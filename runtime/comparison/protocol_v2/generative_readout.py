"""Small, auditable affine readouts for pretrained generator watermarking.

The StyleSwin carrier is the concatenation of *real* ToRGB inputs after all
subsequent linear skip upsamplers.  A single output direction is shared by
all branches.  Consequently its second moment includes cross-branch blocks;
adding branch moments or padding a narrow final layer is not equivalent.

Geometry is native-output geometry: noise for guided diffusion, latent patch
velocity for SiT, and RGB pixels for StyleSwin.  SiT's nonlinear VAE decoder
does not inherit an exact affine RGB-output identity.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from .methods import lerra_insert, lerra_preflight, lerra_signal


def _tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ReadoutBranch:
    module_path: str
    rows: tuple[int, ...] | None = None
    postprocess: Callable | None = None
    transform_id: str = "identity"
    weight_scale: float = 1.0


class AffineReadout:
    """An actual affine weight matrix, optionally assembled from RGB branches.

    ``postprocess`` must be a fixed, channelwise linear spatial map. Factories
    below establish that property for supported public model implementations;
    arbitrary adapters require an independent whole-output identity test.
    """

    def __init__(self, model, branches: Iterable[ReadoutBranch], *, kind="affine"):
        self.model, self.kind = model, kind
        self.branches = tuple(branches)
        if not self.branches or len({b.module_path for b in self.branches}) != len(self.branches):
            raise ValueError("readout branches must be nonempty and distinct")
        self.modules, self.rows, self.widths = [], [], []
        for b in self.branches:
            module = model.get_submodule(b.module_path)
            if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                raise ValueError("readout must be an actual Conv2d or Linear")
            if isinstance(module, torch.nn.Conv2d) and (module.groups != 1 or isinstance(module.padding, str)):
                raise ValueError("grouped and implicit-string-padding convolutions are unsupported")
            if module.weight.dtype != torch.float32:
                raise ValueError("generator insertion requires FP32 master readout weights")
            rows = tuple(range(module.weight.shape[0])) if b.rows is None else b.rows
            if not rows or len(set(rows)) != len(rows) or any(type(i) is not int or i < 0 or i >= module.weight.shape[0] for i in rows):
                raise ValueError("invalid selected output rows")
            if b.weight_scale != 1.0:
                raise ValueError("plain Conv2d/Linear readouts require unit physical weight scale")
            if b.postprocess is not None and (not isinstance(module, torch.nn.Conv2d) or module.kernel_size != (1, 1) or module.stride != (1, 1) or module.padding != (0, 0)):
                raise ValueError("spatial branch transforms require an unpadded pointwise convolution")
            self.modules.append(module)
            self.rows.append(rows)
            self.widths.append(module.weight[0].numel())
        if len({len(rows) for rows in self.rows}) != 1:
            raise ValueError("all branches must share the same output dimension and q")
        if len({id(m.weight) for m in self.modules}) != len(self.modules):
            raise ValueError("aliased parameters cannot inflate the readout dimension")
        if len(self.branches) > 1 and len(set(self.rows)) != 1:
            raise ValueError("joint branches must use identically ordered output channels")
        self.outputs, self.d = len(self.rows[0]), sum(self.widths)

    def contract(self):
        return {"schema": "generative_affine_readout_v1", "kind": self.kind,
                "d": self.d, "outputs": self.outputs,
                "branches": [{"module_path": b.module_path, "weight_shape": list(m.weight.shape),
                              "rows": list(rows), "width": width, "transform": b.transform_id,
                              "weight_scale": b.weight_scale}
                             for b, m, rows, width in zip(self.branches, self.modules, self.rows, self.widths)]}

    def weight_matrix(self):
        return torch.cat([m.weight.detach()[list(rows)].reshape(self.outputs, width)
                          for m, rows, width in zip(self.modules, self.rows, self.widths)], dim=1)

    @torch.no_grad()
    def assign_matrix(self, matrix):
        if matrix.shape != (self.outputs, self.d) or not torch.isfinite(matrix).all():
            raise ValueError("invalid affine readout matrix")
        offset = 0
        for m, rows, width in zip(self.modules, self.rows, self.widths):
            value = matrix[:, offset:offset + width].reshape(len(rows), *m.weight.shape[1:])
            m.weight[list(rows)] = value.to(device=m.weight.device, dtype=m.weight.dtype)
            offset += width

    def capture(self):
        return FeatureCapture(self)


class FeatureCapture(AbstractContextManager):
    """Capture exactly one eval forward, removing hooks even after exceptions."""

    def __init__(self, readout):
        self.readout = readout
        self.inputs, self.calls, self.handles = {}, 0, []
        self.closed = False
        self.successful = False

    def __enter__(self):
        if self.readout.model.training:
            raise ValueError("feature capture requires an eval model")
        if self.handles or self.closed:
            raise ValueError("feature capture contexts cannot be reused")
        def root_hook(_module, _args):
            self.calls += 1
            if self.calls != 1:
                raise ValueError("capture requires exactly one model forward")
        self.handles.append(self.readout.model.register_forward_pre_hook(root_hook))
        for index, module in enumerate(self.readout.modules):
            def hook(_module, args, index=index):
                if index in self.inputs or not args or not torch.is_tensor(args[0]):
                    raise ValueError("each actual readout must execute exactly once")
                # Clone because an upstream implementation may reuse/in-place-edit buffers.
                self.inputs[index] = args[0].detach().clone()
            self.handles.append(module.register_forward_pre_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.closed = True
        if exc_type is None and (self.calls != 1 or len(self.inputs) != len(self.readout.branches)):
            raise ValueError("capture did not observe one forward of every actual readout")
        self.successful = exc_type is None
        return False

    def _layout(self, index):
        if not self.closed or not self.successful or len(self.inputs) != len(self.readout.branches):
            raise ValueError("features require a successfully completed capture")
        x, m, b = self.inputs[index], self.readout.modules[index], self.readout.branches[index]
        if isinstance(m, torch.nn.Linear):
            if x.shape[-1] != m.in_features:
                raise ValueError("invalid linear feature shape")
            return tuple(x.shape[:-1])
        if x.ndim != 4 or x.shape[1] != m.in_channels:
            raise ValueError("invalid convolution feature shape")
        if b.postprocess is not None:
            # Shape probe has one channel, never d full-resolution channels.
            shape = b.postprocess(x[:, :1]).shape
            if len(shape) != 4 or shape[:2] != (x.shape[0], 1):
                raise ValueError("spatial transform must preserve batch and act channelwise")
            return (shape[0], shape[2], shape[3])
        size = [(x.shape[2+i] + 2*m.padding[i] - m.dilation[i]*(m.kernel_size[i]-1) - 1)//m.stride[i] + 1 for i in range(2)]
        return (x.shape[0], *size)

    @property
    def layout(self):
        layouts = [self._layout(i) for i in range(len(self.readout.branches))]
        if any(s != layouts[0] for s in layouts):
            raise ValueError("joint readout branches do not reach the same output coordinates")
        return layouts[0]

    @property
    def count(self):
        return math.prod(self.layout)

    @torch.no_grad()
    def feature_rows(self, indices=None, *, channel_chunk=32):
        """Return selected [observations,d] rows, preserving true cross terms.

        Conv patches are gathered only at selected positions. Joint branches
        propagate a bounded channel chunk through the actual skip operators;
        no [batch,d,256,256] joint feature image is materialized.
        """
        layout = self.layout
        total = math.prod(layout)
        indices = torch.arange(total) if indices is None else torch.as_tensor(indices, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() == 0 or indices.min() < 0 or indices.max() >= total or channel_chunk < 1:
            raise ValueError("invalid selected feature indices or channel chunk")
        result = []
        for index, (m, b) in enumerate(zip(self.readout.modules, self.readout.branches)):
            x = self.inputs[index]
            at = indices.to(x.device)
            if isinstance(m, torch.nn.Linear):
                h = x.reshape(-1, x.shape[-1])[at]
            elif b.postprocess is not None:
                chunks = []
                for start in range(0, x.shape[1], channel_chunk):
                    z = b.postprocess(x[:, start:start + channel_chunk])
                    chunks.append(z.permute(0, 2, 3, 1).reshape(total, z.shape[1])[at])
                h = torch.cat(chunks, dim=1)
            else:
                batch, height, width = layout
                n = torch.div(at, height*width, rounding_mode="floor")
                y = torch.div(at % (height*width), width, rounding_mode="floor") * m.stride[0]
                xx = at % width * m.stride[1]
                ph, pw = m.padding
                padded = F.pad(x, (pw, pw, ph, ph), mode="constant" if m.padding_mode == "zeros" else m.padding_mode)
                # Flatten order is channel,kernel_y,kernel_x, exactly as weight.reshape.
                patches = [padded[n, :, y+ky*m.dilation[0], xx+kx*m.dilation[1]]
                           for ky in range(m.kernel_size[0]) for kx in range(m.kernel_size[1])]
                h = torch.stack(patches, dim=-1).reshape(len(at), -1)
            result.append(h * b.weight_scale)
        return torch.cat(result, dim=1)

    def iter_feature_chunks(self, indices=None, *, rows_per_chunk=256, channel_chunk=32,
                            spatial_rows_per_pass=4096):
        """Stream row chunks without re-running skip maps for every small chunk.

        A spatial pass selects at most 4096 rows by default (40 MiB for d=2560
        in FP32), then yields smaller moment-accumulation chunks. For carrier
        estimation, pass the predetermined sample indices; traversing *all*
        pixels repeats expensive spatial maps once per spatial pass.
        """
        if rows_per_chunk < 1 or not 1 <= spatial_rows_per_pass <= 16384:
            raise ValueError("invalid feature chunk or bounded spatial pass size")
        indices = torch.arange(self.count) if indices is None else torch.as_tensor(indices, dtype=torch.long)
        spatial = any(b.postprocess is not None for b in self.readout.branches)
        pass_size = spatial_rows_per_pass if spatial else rows_per_chunk
        for start in range(0, len(indices), pass_size):
            features = self.feature_rows(indices[start:start + pass_size], channel_chunk=channel_chunk)
            for offset in range(0, len(features), rows_per_chunk):
                yield features[offset:offset + rows_per_chunk]


class StreamingMoment:
    """Exact uncentered second moment in bounded CPU memory (d², not N×d)."""
    def __init__(self, d, *, max_dimension=4096):
        if not isinstance(d, int) or not 1 <= d <= max_dimension:
            raise ValueError("moment dimension outside explicit memory bound")
        self.d, self.count = d, 0
        self.total = torch.zeros(d, d, dtype=torch.float64)

    def update(self, features):
        h = features.detach().to(device="cpu", dtype=torch.float64)
        if h.ndim != 2 or h.shape[1] != self.d or not len(h) or not torch.isfinite(h).all():
            raise ValueError("invalid observed feature rows")
        self.total.add_(h.T @ h)
        self.count += len(h)

    def value(self, *, require_full_sample=True):
        if self.count < (self.d+1 if require_full_sample else 1):
            raise ValueError("insufficient feature observations for declared carrier sample rule")
        return (self.total + self.total.T) / (2*self.count)


def guided_readout(model, path="out.2"):
    m = model.get_submodule(path)
    if not isinstance(m, torch.nn.Conv2d) or tuple(m.weight.shape) != (6, 256, 3, 3):
        raise ValueError("expected public guided-diffusion 256 RGB/variance readout")
    return AffineReadout(model, [ReadoutBranch(path, (0, 1, 2))], kind="guided_noise")


def sit_readout(model, path="final_layer.linear"):
    m = model.get_submodule(path)
    sigma = getattr(model, "learn_sigma", None)
    if not isinstance(m, torch.nn.Linear) or sigma not in (True, False):
        raise ValueError("expected SiT-B/2 4-latent-channel patch-2 velocity readout")
    if sigma is True and tuple(m.weight.shape) == (32, 768):
        # Public SiT-B/2 weights retain the doubled output projection. Native
        # unpatchify orders each patch as (p,p,8), then forward discards the
        # last four channels. Velocity rows are interleaved, NOT the first 16.
        rows = tuple(patch*8 + channel for patch in range(4) for channel in range(4))
    elif sigma is False and tuple(m.weight.shape) == (16, 768):
        rows = tuple(range(16))
    else:
        raise ValueError("SiT learn_sigma metadata and actual projection shape disagree")
    return AffineReadout(model, [ReadoutBranch(path, rows)], kind="sit_latent_velocity")


def styleswin_readout(model, path="to_rgbs"):
    rgbs = model.get_submodule(path)
    widths = [512, 512, 512, 512, 256, 128, 128]
    if len(rgbs) != len(widths):
        raise ValueError("expected seven public StyleSwin-256 ToRGB branches")
    if [getattr(rgb, "is_upsample", None) for rgb in rgbs] != [True]*6 + [False]:
        raise ValueError("unexpected StyleSwin skip topology; last branch must preserve resolution")
    branches = []
    for i, (rgb, width) in enumerate(zip(rgbs, widths)):
        if not isinstance(rgb.conv, torch.nn.Conv2d) or tuple(rgb.conv.weight.shape) != (3, width, 1, 1):
            raise ValueError("unexpected StyleSwin-256 ToRGB width")
        # Official final branch is another 256px contribution, with no
        # ``upsample`` attribute. Its skip edge is the identity, not 2x.
        transform_indices = tuple(j for j in range(i+1, len(rgbs)) if rgbs[j].is_upsample)
        transforms = tuple(rgbs[j].upsample for j in transform_indices)
        if any(t is None for t in transforms):
            raise ValueError("missing subsequent StyleSwin skip upsampler")
        def propagate(x, transforms=transforms):
            for transform in transforms:
                x = transform(x)
            return x
        branches.append(ReadoutBranch(f"{path}.{i}.conv", postprocess=propagate,
                        transform_id="then:" + ",".join(f"{path}.{j}.upsample" for j in transform_indices)))
    return AffineReadout(model, branches, kind="styleswin_joint_rgb")


def make_assets(readout, E, q, *, alpha, tolerance=1e-10):
    """Freeze small native LERRA assets; never copy a full generator."""
    assets = {"schema": "generative_lerra_assets_v1", "readout": readout.contract(),
              "E": E.detach().cpu().double().clone(), "q": q.detach().cpu().double().clone(),
              "W0": readout.weight_matrix().detach().cpu().double(), "alpha": float(alpha),
              "tolerance": float(tolerance), "classification": False}
    lerra_preflight(readout.weight_matrix(), assets)
    return assets


def _check_assets(readout, assets):
    if assets.get("schema") != "generative_lerra_assets_v1" or assets.get("readout") != readout.contract():
        raise ValueError("readout/asset contract mismatch")
    lerra_preflight(readout.weight_matrix(), assets)


def insert(readout, code, assets):
    _check_assets(readout, assets)
    code = torch.as_tensor(code)
    if code.ndim != 1 or not torch.all((code == 0) | (code == 1)):
        raise ValueError("watermark code must be binary")
    matrix = readout.weight_matrix()
    lerra_insert(matrix, code, assets)
    readout.assign_matrix(matrix)
    return extract(readout, assets)


def extract(readout, assets):
    _check_assets(readout, assets)
    return lerra_signal(readout.weight_matrix(), assets).detach().cpu()


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_assets(assets, path):
    if assets.get("schema") != "generative_lerra_assets_v1":
        raise ValueError("unexpected asset schema")
    _atomic_save(assets, path)


def load_assets(readout, path):
    assets = torch.load(path, map_location="cpu", weights_only=True)
    _check_assets(readout, assets)
    return assets


def save_delta(readout, assets, path, *, base_checkpoint_sha256):
    """Save only exact selected FP32 weight differences, with immutable lineage.

    The caller must separately verify the declared full base checkpoint hash
    when loading a model. This utility checks its selected weights and adapter.
    """
    _check_assets(readout, assets)
    if len(base_checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in base_checkpoint_sha256):
        raise ValueError("a verified lowercase SHA256 base checkpoint identity is required")
    current = readout.weight_matrix().cpu()
    origin = assets["W0"].float()
    payload = {"schema": "generative_readout_delta_v1", "readout": readout.contract(),
               "base_checkpoint_sha256": base_checkpoint_sha256,
               "base_selected_sha256": _tensor_hash(origin), "result_selected_sha256": _tensor_hash(current),
               "delta": current.double() - origin.double()}
    _atomic_save(payload, path)
    return payload


def apply_delta(readout, path, *, base_checkpoint_sha256):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "generative_readout_delta_v1" or payload.get("readout") != readout.contract() or payload.get("base_checkpoint_sha256") != base_checkpoint_sha256:
        raise ValueError("delta/checkpoint/adapter identity mismatch")
    current = readout.weight_matrix().cpu()
    digest = _tensor_hash(current)
    if digest == payload["result_selected_sha256"]:
        return {"applied": False, "already_applied": True}
    if digest != payload["base_selected_sha256"]:
        raise ValueError("delta cannot be applied to changed base readout weights")
    target = (current.double() + payload["delta"]).float()
    if _tensor_hash(target) != payload["result_selected_sha256"]:
        raise ValueError("delta reconstruction hash mismatch")
    readout.assign_matrix(target)
    return {"applied": True, "already_applied": False}
