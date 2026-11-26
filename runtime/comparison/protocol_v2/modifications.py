import copy
import math
import torch


def parameter_change(original,modified,names):
    """Actual parameter change; buffers and nominal sparsity are not conflated."""
    squared,maximum,changed,total=0.,0.,0,0
    for name in names:
        a,b=original[name].detach().cpu(),modified[name].detach().cpu()
        if a.shape!=b.shape: raise ValueError("parameter shape changed during modification")
        d=b.double()-a.double()
        squared+=float(d.square().sum())
        maximum=max(maximum,float(d.abs().max()))
        changed+=int(torch.count_nonzero(d))
        total+=d.numel()
    return {"l2":squared**.5,"linf":maximum,"changed_coordinates":changed,"parameter_coordinates":total,"scope":"named_parameters; buffers excluded"}


def collude(states, layers):
    """Attacker knows participants; decoder receives only returned suspect state."""
    if len(states) not in (2, 3):
        raise ValueError("only 2/3-person uniform collusion is in scope")
    host = min(states)
    out = copy.deepcopy(states[host])
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("predeclared whole tensor list required")
    for n in layers:
        if not n.endswith("weight"):
            raise ValueError("collusion must average declared whole weight tensors")
        xs = [states[i][n] for i in sorted(states)]
        if any(x.shape != xs[0].shape or x.dtype != xs[0].dtype for x in xs):
            raise ValueError("incompatible collusion tensors")
        out[n] = torch.stack([x.double() for x in xs]).mean(0).to(xs[0])
    return out, host


def modify(state, spec):
    out = copy.deepcopy(state)
    kind = spec["kind"]
    receipt = {"kind": kind, "tensors": {}}
    if kind == "none":
        return out, receipt
    if kind not in ("pruning", "qdq"):
        raise ValueError("FT uses training engine; unknown nonlearning modification")
    names = spec["layers"]
    if not names or len(set(names)) != len(names):
        raise ValueError("explicit unique modification tensor list required")
    for n in names:
        if n not in out or not torch.isfinite(out[n]).all():
            raise ValueError("invalid modification tensor")
    if kind == "pruning":
        if spec["algorithm"] != "global_magnitude_stable_index" or not 0 <= spec["fraction"] <= 1:
            raise ValueError("unsupported pruning rule")
        flat = torch.cat([out[n].reshape(-1) for n in names])
        count = math.floor(flat.numel()*spec["fraction"])
        mask = torch.ones_like(flat, dtype=torch.bool)
        mask[torch.argsort(flat.abs(), stable=True)[:count]] = False
        cursor = 0
        for n in names:
            v = mask[cursor:cursor+out[n].numel()].reshape_as(out[n])
            out[n] *= v
            receipt["tensors"][n] = {"mask": v.cpu(), "removed": int((~v).sum())}
            cursor += out[n].numel()
    else:
        if spec["algorithm"] != "symmetric_per_tensor_nearest_even" or not 2 <= spec["bits"] <= 16:
            raise ValueError("unsupported quantization rule")
        qmax = 2**(spec["bits"]-1)-1
        for n in names:
            w = out[n]
            scale = w.abs().max().double()/qmax
            if scale > 0:
                out[n] = (torch.round(w.double()/scale).clamp(-qmax,qmax)*scale).to(w)
            receipt["tensors"][n] = {"scale": float(scale), "zero_tensor": bool(scale == 0), "bits": spec["bits"]}
    return out, receipt
