"""Versioned last-SiT-MLP LessMark carrier; frozen protocol sources stay intact.

For delta W = alpha q (E b)^T, the observed first-order native velocity
energy is alpha^2 b^T E^T M E b, where
M = mean(||J_velocity(x) diag(gate) q||^2 h h^T).
Low eigenvectors minimize its trace for isotropic payload signs. This is a
local, clean-model native-velocity objective, not the CFG/ODE/VAE RGB Jacobian
and not a global optimum of RGB MSE. Actual final RGB must be validated by the
caller on the declared validation partition before selecting alpha.
"""
from __future__ import annotations

import math

import torch
from torch import nn

LAYER = "blocks.11.mlp.fc2"
PAYLOAD_BITS = 256
Q_SEED = 710101
ACTIVE_ROWS = tuple(patch * 8 + channel for patch in range(4) for channel in range(4))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def balanced_q(outputs, seed=Q_SEED):
    require(type(outputs) is int and outputs > 0 and outputs % 2 == 0,
            "balanced q requires a positive even output dimension")
    require(type(seed) is int and seed >= 0, "invalid q seed")
    signs = torch.cat((torch.ones(outputs // 2, dtype=torch.float64),
                       -torch.ones(outputs // 2, dtype=torch.float64)))
    order = torch.randperm(outputs, generator=torch.Generator(device="cpu").manual_seed(seed))
    return signs[order] / math.sqrt(outputs)


def final_velocity_jq(x, gate, q, scale, final_weight, eps, active_rows=ACTIVE_ROWS):
    """Exact analytic differential of the frozen tokenwise final SiT tail.

    x is the actual post-last-block input of final LayerNorm, [B,T,D].
    gate [B,D] is last-block gate_mlp; scale [B,D] is final AdaLN's scale
    chunk (not 1+scale); q [D]. Returns [B,T,len(active_rows)] in FP64.
    The omitted final shift and linear bias have zero differential.
    """
    x, gate, q, scale, weight = [t.detach().to(dtype=torch.float64)
                                for t in (x, gate, q, scale, final_weight)]
    require(x.ndim == 3 and gate.shape == scale.shape == (x.shape[0], x.shape[2])
            and q.shape == (x.shape[2],) and weight.ndim == 2
            and weight.shape[1] == x.shape[2], "invalid final-tail differential shapes")
    require(all(t.device == x.device for t in (gate, q, scale, weight)),
            "differential tensors must share a device")
    rows = tuple(active_rows)
    require(rows and len(rows) == len(set(rows)) and all(type(i) is int and 0 <= i < len(weight) for i in rows),
            "invalid active native velocity rows")
    require(math.isfinite(eps) and eps > 0 and all(bool(torch.isfinite(t).all()) for t in (x, gate, q, scale, weight)),
            "nonfinite tail inputs or invalid LayerNorm epsilon")
    centered = x - x.mean(dim=-1, keepdim=True)
    inverse_std = (centered.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    direction = (gate * q).unsqueeze(1)
    differential = (inverse_std * (direction - direction.mean(dim=-1, keepdim=True))
                    - inverse_std.pow(3) * centered
                    * (centered * direction).mean(dim=-1, keepdim=True))
    return (differential * (1 + scale).unsqueeze(1)) @ weight[list(rows)].T


def low_basis(moment, bits=PAYLOAD_BITS):
    """Deterministic sign convention; full symmetric CPU eigendecomposition."""
    m = moment.detach().to(device="cpu", dtype=torch.float64)
    require(m.ndim == 2 and m.shape[0] == m.shape[1] and 1 <= bits and 2 * bits <= len(m)
            and bool(torch.isfinite(m).all()), "invalid internal carrier capacity or moment")
    m = (m + m.T) / 2
    values, vectors = torch.linalg.eigh(m)
    scale = float(values.abs().max())
    require(scale > 0 and float(values[0]) >= -max(scale * 1e-9, 1e-15),
            "weighted moment is zero or not numerically positive semidefinite")
    pivots = vectors.abs().argmax(dim=0)
    vectors *= torch.sign(vectors[pivots, torch.arange(len(values))])
    return {"moment": m, "eigenvalues": values, "eigenvectors": vectors,
            "bases": {f"{bits}/Low": vectors[:, :bits].clone()}}


def _architecture(backend):
    require(backend.kind == "sit_b_2", "internal carrier is specific to public SiT-B/2")
    model = backend.model
    require(not model.training and len(model.blocks) == 12 and model.learn_sigma is True
            and model.patch_size == 2 and model.out_channels == 8,
            "expected frozen eval SiT-B/2 with the declared final tokenwise tail")
    layer = model.get_submodule(LAYER)
    final = model.final_layer
    require(isinstance(layer, nn.Linear) and tuple(layer.weight.shape) == (768, 3072)
            and isinstance(final.linear, nn.Linear) and tuple(final.linear.weight.shape) == (32, 768)
            and isinstance(final.norm_final, nn.LayerNorm)
            and tuple(final.norm_final.normalized_shape) == (768,)
            and final.norm_final.elementwise_affine is False,
            "unexpected internal Linear or final head architecture")
    require(all(not isinstance(m, nn.Dropout) or m.p == 0 for m in model.blocks[-1].mlp.modules()),
            "analytic final-tail contract excludes post-fc2 dropout")
    require(layer.weight.dtype == torch.float32 and bool(torch.isfinite(layer.weight).all()),
            "internal carrier requires finite FP32 master weights")
    return layer, final


def _hooks(backend, captured):
    model = backend.model
    def capture(name, pre=False):
        def hook(module, args, output=None):
            captured[name].append((args[0] if pre else output).detach())
        return hook
    return [model.get_submodule(LAYER).register_forward_pre_hook(capture("h", True)),
            model.blocks[-1].adaLN_modulation.register_forward_hook(capture("gate")),
            model.final_layer.norm_final.register_forward_pre_hook(capture("x", True)),
            model.final_layer.adaLN_modulation.register_forward_hook(capture("scale"))]
