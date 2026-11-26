"""Native operations and declared BIBD adaptations. No decoder accepts truth."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
from .codebooks import Decision

METHODS = ("lerra", "uchida", "greedy", "neuralmark", "deepmarks", "tattooed")


@dataclass
class Recovery:
    bits: np.ndarray | None
    status: str
    signal: np.ndarray | None = None
    details: dict = field(default_factory=dict)


def decode(signal, threshold=0., erasure_at_zero=False):
    z = signal.detach().double().cpu().numpy() if torch.is_tensor(signal) else np.asarray(signal)
    if z.ndim != 1 or not np.isfinite(z).all():
        return Recovery(None, "invalid", details={"reason": "nonfinite or invalid signal shape","invalid_coordinates":int((~np.isfinite(z)).sum()) if np.issubdtype(z.dtype,np.number) else None})
    if erasure_at_zero and np.any(z == 0):
        return Recovery(None, "erasure", z, {"erasure_coordinates": int((z == 0).sum())})
    return Recovery((z > threshold).astype(np.uint8), "ok", z)


def tensor(model, name):
    p = dict(model.named_parameters())
    if name not in p:
        raise ValueError(f"missing declared embedding tensor: {name}")
    return p[name]


def weight_matrix(w, rows=None):
    w = w if rows is None else w[:rows]
    return w.reshape(w.shape[0], -1)


def orthonormal(x):
    q, _ = torch.linalg.qr(x.double(), mode="reduced")
    at = q.abs().argmax(dim=0)
    signs = torch.sign(q[at, torch.arange(q.shape[1], device=q.device)])
    return q * signs


def lerra_preflight(w, assets):
    e, q, w0 = assets["E"].double(), assets["q"].double(), assets["W0"].double()
    mat = weight_matrix(w, assets.get("rows"))
    d, k = e.shape
    tol = assets["tolerance"]
    if k < 1 or 2*k > d or mat.shape != (q.numel(), d) or w0.shape != mat.shape:
        raise ValueError("LERRA actual d/2, q, E, W0 or tensor shape violation")
    if not all(torch.isfinite(x).all() for x in (e, q, w0, mat)):
        raise ValueError("nonfinite LERRA asset")
    if not torch.allclose(e.T @ e, torch.eye(k, device=e.device, dtype=e.dtype), atol=tol, rtol=0):
        raise ValueError("LERRA E is not orthonormal")
    if abs(float(q.norm())-1) > tol or not math.isfinite(assets["alpha"]) or assets["alpha"] <= 0:
        raise ValueError("LERRA q/positive alpha invalid")
    if assets.get("classification") and abs(float(q.sum())) > tol:
        raise ValueError("classification q must have zero class mean")
    return {"d": d, "k": k, "weight_shape": list(w.shape), "norm_target": assets["alpha"]*math.sqrt(k)}


def lerra_signal(w, assets):
    w = weight_matrix(w, assets.get("rows")).double()
    e, q, w0 = (assets[n].to(w.device, dtype=torch.float64) for n in ("E", "q", "W0"))
    return e.T @ (w - w0).T @ q


def lerra_insert(w, code, assets):
    lerra_preflight(w, assets)
    mat = weight_matrix(w, assets.get("rows"))
    e, q, w0 = (assets[n].to(w.device, dtype=torch.float64) for n in ("E", "q", "W0"))
    b = torch.as_tensor(code, device=w.device, dtype=torch.float64)*2-1
    if b.shape != (e.shape[1],):
        raise ValueError("LERRA payload shape mismatch")
    with torch.no_grad():
        mat.copy_((w0 + assets["alpha"] * q[:, None] * (e @ b)[None, :]).to(w.dtype))


def lerra_attribution(z, codes, *, alpha, cutoff, tie_atol=1e-12, block_size=4096):
    """R6 method-gate equations; preserve strict-rank and undefined state separately.

    R6 runner.py:1062-1090: top coefficient >= alpha/2, cosine >= cutoff,
    top-second coefficient > 1e-12. Undefined/tie are not normal rejections.
    Cutoff provenance must be checked by the caller; no M correction here.
    """
    z = np.asarray(z, dtype=np.float64)
    k = codes.shape[1]
    if z.shape != (k,) or not np.isfinite(z).all():
        return Decision("invalid", details={"native_gate_passed": False, "reason": "nonfinite signal"})
    norm = np.linalg.norm(z)
    top = []
    passes = 0
    for start in range(0, len(codes), block_size):
        signs = 2*np.asarray(codes[start:start+block_size], dtype=np.float64)-1
        co = signs @ z / k
        cs = co * math.sqrt(k)/norm if norm != 0 else np.full(len(co), np.nan)
        passes += int(((co >= alpha/2) & (cs >= cutoff)).sum())
        top.extend((float(co[i]), int(start+i), float(cs[i])) for i in np.argsort(-co, kind="stable")[:2])
        top = sorted(top, key=lambda x: (-x[0], x[1]))[:2]
    if len(top) < 2:
        raise ValueError("all-key comparison requires at least two registered codes")
    gap = top[0][0]-top[1][0]
    info = {"candidate_threshold_passes": passes, "coefficient": top[0][0], "cosine": top[0][2] if norm else None, "gap": gap, "native_gate_passed": False, "native_accepted_key": None, "strict_rank_state": "tie" if gap <= tie_atol else "unique"}
    if norm == 0:
        return Decision("undefined", details={**info, "reason": "zero signal: cosine undefined; native gate false, strict-rank tie"})
    if gap <= tie_atol:
        return Decision("tie", details=info)
    passed = top[0][0] >= alpha/2 and top[0][2] >= cutoff
    info.update(native_gate_passed=bool(passed), native_accepted_key=top[0][1] if passed else None)
    return Decision("id" if passed else "no_match", recipient=top[0][1] if passed else None, details=info)


def uchida_vector(w):
    # Keras HWIO -> mean(O) -> HWI. Matches prior R6 PyTorch port.
    if w.ndim == 4:
        return w.mean(0).permute(1, 2, 0).reshape(-1)
    if w.ndim == 2:
        return w.mean(0)
    raise ValueError("Uchida expects Conv2d or Linear weight")


def greedy_residual(weights, bits=256, width=16, divider=10.):
    rows = []
    for w in weights:
        a = w.reshape(-1)[:w.numel()//width*width]
        if not a.numel():
            raise ValueError("Greedy tensor too small")
        # CUDA adaptive-pool backward lacks deterministic support in PyTorch
        # 2.5. Retain the native bins/autograd through explicit CPU transfers.
        pooled_input=a.cpu() if a.is_cuda and torch.are_deterministic_algorithms_enabled() else a
        rows.append(F.adaptive_avg_pool1d(pooled_input[None, None], bits*width).reshape(bits, width).to(a.device))
    a = torch.stack(rows).mean(0)
    n = int(width/divider + .5) if divider else 0
    if n < 0 or n > width:
        raise ValueError("invalid Greedy divider")
    if n:
        indices = torch.argsort(a.abs(), dim=1)[:, :n]
        a = a.scatter(1, indices, 0.)
    return a.mean(1)


def neural_filter(w, filter_bits, pooled=512):
    f = filter_bits.to(w.device).bool()
    for _ in range(4):
        w = w.reshape(-1)[:w.numel()//f.numel()*f.numel()]
        w = w[f.repeat(w.numel()//f.numel())]
    if w.numel() < pooled:
        raise ValueError("NeuralMark filtered weight length <512; no key resampling")
    return w[:w.numel()//pooled*pooled].reshape(-1, pooled).mean(0)


def deepmarks_validate(w, assets):
    x, u = assets["X"].double(), assets["U"].double()
    d = w[0].numel()
    k = u.shape[1]
    if x.shape != (u.shape[0], d) or u.shape[0] < k or d < k:
        raise ValueError("DeepMarks projection/layer dimensions invalid")
    if not torch.isfinite(x).all() or not torch.isfinite(u).all():
        raise ValueError("nonfinite DeepMarks assets")
    if not torch.allclose(u.T@u, torch.eye(k, dtype=u.dtype), atol=assets["tolerance"], rtol=0):
        raise ValueError("DeepMarks U^T U != I")
    rank = int(torch.linalg.matrix_rank(u.T @ x))
    if rank < k:
        raise ValueError("DeepMarks effective projection rank < payload")
    if assets["reduction"] not in ("mean", "sum"):
        raise ValueError("DeepMarks MSE reduction must be predeclared")
    return {"X_shape": list(x.shape), "U_shape": list(u.shape), "effective_rank": rank, "layer_dimension": d, "reduction": assets["reduction"], "target_coefficients": [-1, 1], "normalization": "none", "decoder": ">0.85", "extension": "original threshold retained at 256 coordinates"}


class Method:
    def __init__(self, name, assets):
        if name not in METHODS:
            raise ValueError(name)
        self.name, self.assets = name, assets

    def signal(self, model):
        a = self.assets
        w = tensor(model, a["layers"][0])
        if self.name == "lerra":
            return lerra_signal(w, a)
        if self.name == "uchida":
            return uchida_vector(w) @ a["P"].to(w)
        if self.name == "greedy":
            return greedy_residual([tensor(model, n) for n in a["layers"]], a["bits"], a["width"], a["divider"])
        if self.name == "neuralmark":
            return neural_filter(w, a["filter_bits"], a["P"].shape[0]) @ a["P"].to(w)
        if self.name == "deepmarks":
            return a["U"].to(w).T @ a["X"].to(w) @ w.mean(0).reshape(-1)
        raise ValueError("TATTOOED has an LDPC recovery object, not this linear signal API")

    def loss(self, model, target):
        a = self.assets
        w = tensor(model, a["layers"][0])
        c = torch.as_tensor(target, device=w.device, dtype=w.dtype)
        if self.name == "deepmarks":
            f = a["U"].to(w) @ (2*c-1)
            return F.mse_loss(a["X"].to(w) @ w.mean(0).reshape(-1), f, reduction=a["reduction"])
        z = self.signal(model)
        if self.name == "uchida":
            return F.binary_cross_entropy_with_logits(z, c, reduction="sum")
        if self.name == "neuralmark":
            return F.binary_cross_entropy_with_logits(z, c, reduction="mean")
        if self.name == "greedy":
            return F.relu(a["margin"]-(2*c-1)*z).sum()
        raise ValueError("direct methods have no insertion optimizer loss")

    def recover(self, model, collusion=False):
        if self.name == "tattooed":
            from .tattooed import extract
            return extract(model, self.assets)
        z = self.signal(model)
        if self.name == "lerra" and collusion:
            return decode(z/self.assets["alpha"], 2/3)
        if self.name in ("uchida", "neuralmark"):
            if not torch.isfinite(z).all():
                return decode(z)  # sigmoid(+/-inf) must not hide invalid extraction.
            result = decode(torch.sigmoid(z), .5)
            result.signal = z.detach().double().cpu().numpy()
            return result
        return decode(z, .85 if self.name == "deepmarks" else 0., self.name in ("lerra", "greedy"))

    def insert(self, model, target):
        if self.name == "lerra":
            lerra_insert(tensor(model, self.assets["layers"][0]), target, self.assets)
        elif self.name == "tattooed":
            from .tattooed import insert
            insert(model, target, self.assets)
        else:
            raise ValueError("learning-based method requires joint training")


def make_assets(name, model, settings, seed, bits=256):
    a = dict(settings)
    g = torch.Generator().manual_seed(seed)
    w = tensor(model, a["layers"][0]).detach().cpu()
    a["bits"] = bits
    if name=="greedy":
        a["deterministic_pool_backend"]="native CPU adaptive pooling with differentiable device copies when CUDA determinism is required"
    if name == "uchida":
        a["P"] = torch.randn(uchida_vector(w).numel(), bits, generator=g)
    elif name == "neuralmark":
        a["P"] = torch.randn(512, bits, generator=g)*a["projection_std"]
        raw = hashlib.shake_256(a["P"].numpy().tobytes()).digest(bits//8)
        a["filter_bits"] = torch.from_numpy(np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).copy())
        neural_filter(w, a["filter_bits"])
    elif name == "deepmarks":
        a["U"] = orthonormal(torch.randn(bits, bits, generator=g, dtype=torch.float64))
        a["X"] = torch.randn(bits, w[0].numel(), generator=g, dtype=torch.float64)
        a["validation"] = deepmarks_validate(w, a)
    elif name == "tattooed":
        from .tattooed import prepare
        a = prepare(model, a, bits)
    elif name not in ("lerra", "greedy"):
        raise ValueError(name)
    return a


def _greedy_signature(seed):
    """Native RSA512/SHA256 signature pipeline with reproducible key entropy.

    The upstream implementation uses OS entropy. This experiment replaces only
    its entropy stream so a declared seed fixes one copyright signature before
    any candidate is observed. The PKCS#1 v1.5 SHA256 operation is unchanged.
    """
    entropy = hashlib.shake_256(f'greedy-residuals/native-key/{seed}'.encode())
    stream = entropy.digest(262144)
    cursor = 0
    def prime():
        nonlocal cursor
        while True:
            if cursor+32 > len(stream):
                raise RuntimeError('predeclared Greedy RSA entropy exhausted')
            n = int.from_bytes(stream[cursor:cursor+32], 'big') | (3 << 254) | 1
            cursor += 32
            if (n-1) % 65537 == 0 or any(n % p == 0 for p in (3,5,7,11,13,17,19,23,29,31,37,41,43,47)):
                continue
            d, power = n-1, 0
            while d % 2 == 0:
                d //= 2; power += 1
            probable = True
            for a in (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53):
                x = pow(a,d,n)
                if x in (1,n-1):
                    continue
                for _ in range(power-1):
                    x = x*x % n
                    if x == n-1:
                        break
                else:
                    probable = False
                    break
            if probable:
                return n
    p, q = prime(), prime()
    while q == p:
        q = prime()
    modulus, exponent = p*q, 65537
    private = pow(exponent,-1,(p-1)*(q-1))
    copyright_message = b'This work is done by Peking University'
    digest_info = bytes.fromhex('3031300d060960864801650304020105000420')+hashlib.sha256(copyright_message).digest()
    size = (modulus.bit_length()+7)//8
    encoded = b'\x00\x01'+b'\xff'*(size-len(digest_info)-3)+b'\x00'+digest_info
    signature = pow(int.from_bytes(encoded,'big'),private,modulus).to_bytes(size,'big')
    if pow(int.from_bytes(signature,'big'),exponent,modulus) != int.from_bytes(encoded,'big'):
        raise RuntimeError('Greedy RSA signature verification failed')
    return hashlib.sha256(signature).digest(), {'copyright_message':copyright_message.decode(), 'rsa_bits':512,
        'public_modulus_hex':hex(modulus),'public_exponent':exponent,'signature_sha256':hashlib.sha256(signature).hexdigest(),
        'entropy_rule':'SHAKE256 seeded local entropy; native RSA512 PKCS1v1.5 SHA256 then signature SHA256'}


def native_payload(name, seed=100, bits=256, *, assets=None):
    """One method-native final message, independent of recipient attribution.

    Internal LDPC/ECC expansion is performed by the method's insertion code;
    this function always returns the final 256-bit target, never encoded bits.
    """
    if name not in METHODS or bits != 256:
        raise ValueError('native comparison payloads require a known method and 256 final message bits')
    details = {'method':name,'seed':seed,'message_bits':bits,'recipient_registration':False}
    if name == 'uchida':
        code = np.ones(bits,dtype=np.uint8)
        details['rule'] = 'native train_wrn.py all-one signature; projection key varies with seed'
    elif name == 'greedy':
        digest, extra = _greedy_signature(seed)
        code = np.unpackbits(np.frombuffer(digest,dtype=np.uint8)).copy()
        details.update(rule='native copyright RSA signature followed by SHA256', **extra)
    elif name == 'neuralmark':
        if assets is None or 'P' not in assets:
            raise ValueError('NeuralMark native signature requires its actual projection key P')
        projection = assets['P'].detach().cpu().contiguous().numpy()
        raw = projection.tobytes()
        code = np.unpackbits(np.frombuffer(hashlib.shake_256(raw).digest(bits//8),dtype=np.uint8)).copy()
        details.update(rule='native ks_generator SHAKE256(actual projection key bytes)', projection_sha256=hashlib.sha256(raw).hexdigest())
    elif name == 'deepmarks':
        from .codebooks import bibd256
        codes, _ = bibd256()
        row = int(np.random.RandomState(seed).randint(len(codes)))
        code = codes[row].copy()
        details.update(rule='one native BIBD anti-collusion fingerprint; no registry search', native_fingerprint_row=row)
    else:
        code = np.random.RandomState(seed).randint(0,2,bits).astype(np.uint8)
        details['rule'] = ('native external binary message then LDPC encoding' if name == 'tattooed'
                           else 'native random binary/sign payload')
    details['message_sha256'] = hashlib.sha256(code.tobytes()).hexdigest()
    return code, details
