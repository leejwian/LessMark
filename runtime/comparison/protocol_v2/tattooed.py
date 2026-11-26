"""Port of pagiux/tattooed 1de12ce; target-free extraction and local RNG.

Preserves replacement-index last-assignment semantics, LDPC/preamble/gain/SNR.
Returns decoded bits (including nonconvergence), never repairs using target.
"""
import warnings
import numpy as np
import torch
from .methods import Recovery


def prepare(model, settings, bits):
    from pyldpc import make_ldpc
    a = dict(settings)
    state = model.state_dict()
    if not a.get("layers"):
        raise ValueError("TATTOOED whole weight tensor list must be fixed before execution")
    if any(n not in state or not n.endswith("weight") for n in a["layers"]):
        raise ValueError("invalid TATTOOED tensor list")
    a["shapes"] = {n: list(state[n].shape) for n in a["layers"]}
    a["H"], a["G"] = make_ldpc(384, 3, 12, systematic=True, sparse=True, seed=a["ldpc_seed"])
    a["preamble"] = np.sign(np.random.RandomState(a["ldpc_seed"]).uniform(-1, 1, 200))
    a["bits"] = bits
    a["encoded_length"] = 200 + (bits // a["G"].shape[1]+1)*384
    n = sum(state[n].numel() for n in a["layers"])
    a["positions"] = np.random.RandomState(a["parameter_seed"]).randint(0, n, int(n*a["ratio"]), np.int32)
    if len(a["positions"]) < 1:
        raise ValueError("TATTOOED has no selected weights")
    return a


def flatten(model, a):
    state = model.state_dict()
    for n, shape in a["shapes"].items():
        if list(state[n].shape) != shape:
            raise ValueError("TATTOOED tensor shape changed")
    return np.concatenate([state[n].detach().cpu().numpy().reshape(-1) for n in a["layers"]])


def encode_message(code, a):
    from pyldpc import encode
    c = np.asarray(code, dtype=np.uint8)
    if c.shape != (a["bits"],) or not np.isin(c, [0, 1]).all():
        raise ValueError("TATTOOED payload invalid")
    k = a["G"].shape[1]
    symbols = []
    rng = np.random.RandomState(a["ldpc_seed"])
    for j in range(len(c)//k+1):
        msg = np.zeros(k, dtype=np.int64)
        part = c[j*k:(j+1)*k]
        msg[:len(part)] = part
        symbols.extend(encode(a["G"], msg, 1e16, seed=rng))
    return np.concatenate((a["preamble"], symbols))


def insert(model, target, a):
    w = flatten(model, a)
    pos = a["positions"]
    rng = np.random.RandomState(a["spreading_seed"])
    for b in encode_message(target, a):
        spread = rng.choice([-1, 1], size=len(pos))
        # Do not use np.add.at: upstream repeated indices are last-write-wins.
        w[pos] = w[pos] + a["gamma"]*spread*b
    state, cursor = model.state_dict(), 0
    with torch.no_grad():
        for n in a["layers"]:
            count = state[n].numel()
            state[n].copy_(torch.from_numpy(w[cursor:cursor+count].reshape(a["shapes"][n])).to(state[n]))
            cursor += count


def extract(model, a):
    from pyldpc import decode, get_message
    w = flatten(model, a)[a["positions"]]
    if not np.isfinite(w).all():
        return Recovery(None, "invalid", details={"reason": "nonfinite weights"})
    rng = np.random.RandomState(a["spreading_seed"])
    y = np.array([rng.choice([-1, 1], size=len(w)) @ w for _ in range(a["encoded_length"])])
    gain = float(np.mean(y[:200]*a["preamble"]))
    if not np.isfinite(gain) or gain == 0:
        return Recovery(None, "decode_failure", y, {"reason": "undefined gain"})
    sigma = float(np.std(y[:200]*a["preamble"]/gain))
    if not np.isfinite(sigma) or sigma <= 0:
        return Recovery(None, "decode_failure", y, {"reason": "undefined SNR"})
    snr = float(-20*np.log10(sigma))
    bits, converged = [], True
    with warnings.catch_warnings(record=True) as caught:
        for j in range(200, len(y), 384):
            d = decode(a["H"], y[j:j+384]/gain, snr, maxiter=a["maxiter"])
            converged &= bool(np.all(a["H"] @ d % 2 == 0))
            bits.extend(get_message(a["G"], d))
    return Recovery(np.asarray(bits[:a["bits"]], dtype=np.uint8), "ok" if converged else "decode_nonconvergence", y, {"gain": gain, "snr": snr, "parity_valid": converged, "warnings": len(caught), "encoded_length": len(y), "message_length": a["bits"]})
