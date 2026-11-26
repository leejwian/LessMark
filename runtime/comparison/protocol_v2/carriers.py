import math
import numpy as np
import torch
import torch.nn.functional as F
from .methods import orthonormal


def q_vector(outputs, seed, classification):
    q = torch.randn(outputs, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    if classification:
        q -= q.mean()
    if q.norm() == 0:
        raise ValueError("zero output direction")
    return q/q.norm()


def carriers(moment, ks, random_seeds):
    u = (moment.double()+moment.double().T)/2
    if not torch.isfinite(u).all() or any(k < 1 or 2*k > u.shape[0] for k in ks):
        raise ValueError("moment or payload dimension invalid")
    values, vectors = torch.linalg.eigh(u)
    at = vectors.abs().argmax(0)
    vectors *= torch.sign(vectors[at, torch.arange(len(values), device=u.device)])
    rs = [orthonormal(torch.randn(len(values), max(ks), dtype=torch.float64, generator=torch.Generator().manual_seed(s))) for s in random_seeds]
    bases = {}
    for k in ks:
        bases[f"{k}/Low"] = vectors[:, :k].cpu()
        bases[f"{k}/High"] = vectors[:, -k:].cpu()
        for i, r in enumerate(rs):
            bases[f"{k}/Random{i}"] = r[:, :k]
    return {"moment": u.cpu(), "eigenvalues": values.cpu(), "eigenvectors": vectors.cpu(), "bases": bases}


@torch.no_grad()
def classification_moment(model, module, batches, expected_images, device):
    model.eval()
    captured = []
    handle = module.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
    count, moment = 0, None
    try:
        for x, _ in batches:
            captured.clear()
            model(x.to(device))
            if len(captured) != 1 or captured[0].ndim != 2:
                raise ValueError("readout input hook did not capture one actual Linear input")
            h = captured[0].double()
            moment = h.T@h if moment is None else moment+h.T@h
            count += len(h)
    finally:
        handle.remove()
    if count != expected_images or count < moment.shape[0]+1:
        raise ValueError("classification carrier sample rule violated: N=min(16384,N_train), N>=d+1")
    return ((moment+moment.T)/(2*count)).cpu()


def proportional_sample(labels, n, seed):
    y = np.asarray(labels)
    classes, counts = np.unique(y, return_counts=True)
    quota = counts*n/len(y)
    take = np.floor(quota).astype(int)
    order = sorted(range(len(classes)), key=lambda i: (-(quota[i]-take[i]), classes[i]))
    for i in order[:n-int(take.sum())]:
        take[i] += 1
    rng = np.random.default_rng(seed)
    ids = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:n] for c,n in zip(classes,take)])
    return rng.permutation(ids).tolist()


def geometry(moment, bases):
    out = {}
    for name, e in bases.items():
        projected = e.T@moment.double()@e
        out[name] = {"lambda_max": float(torch.linalg.eigvalsh(projected)[-1]), "trace": float(torch.trace(projected)), "orthogonality_error": float((e.T@e-torch.eye(e.shape[1])).abs().max())}
    for k in {int(x.split('/')[0]) for x in bases}:
        out[f"{k}/LowHigh_overlap"] = float(torch.linalg.matrix_norm(bases[f"{k}/Low"].T@bases[f"{k}/High"])**2)
    return out
