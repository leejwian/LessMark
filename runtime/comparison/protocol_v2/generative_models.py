"""Pinned public-weight generators for the isolated 256-bit implementation.

This module never downloads weights, trains a model, changes a production run,
loads Python pickles with ``weights_only=False``, or substitutes classifier
outputs for RGB images. ``forward`` exposes each model's native output;
``sample_rgb`` completes its native sampling/decoding path and returns RGB [0,1].
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from .generative_conditions import conditioning, require_reference_contract


class GeneratorContractError(ValueError):
    """An asset, architecture, or output violates the declared native model."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(binding: Mapping[str, Any], *, root: Path | None = None) -> Path:
    if not isinstance(binding, Mapping) or not {'path', 'sha256'} <= binding.keys():
        raise GeneratorContractError('file binding requires path and sha256')
    path = Path(binding['path'])
    if root is not None and not path.is_absolute():
        path = root / path
    if path.is_symlink() or not path.is_file():
        raise GeneratorContractError(f'bound regular file missing: {path}')
    if root is not None and not path.resolve().is_relative_to(root.resolve()):
        raise GeneratorContractError('bound source file escapes source root')
    digest = binding['sha256']
    if not isinstance(digest, str) or len(digest) != 64:
        raise GeneratorContractError('invalid sha256 binding')
    if 'bytes' in binding and path.stat().st_size != binding['bytes']:
        raise GeneratorContractError(f'bound file size changed: {path}')
    if _sha256(path) != digest:
        raise GeneratorContractError(f'bound file hash changed: {path}')
    return path.resolve()


def verify_source(source: Mapping[str, Any]) -> Path:
    """Verify full Python inventory and the pinned revision, before any import."""
    root = Path(source['path']).resolve()
    revision = source.get('revision')
    if not isinstance(revision, str) or len(revision) != 40:
        raise GeneratorContractError('source requires a full immutable commit revision')
    files = source.get('files')
    if not isinstance(files, Mapping) or not files:
        raise GeneratorContractError('source requires a complete file hash inventory')
    bound = set()
    for relative, item in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise GeneratorContractError('source inventory paths must stay relative')
        binding = {'path': str(root / relative_path), 'sha256': item} if isinstance(item, str) else dict(item)
        binding.setdefault('path', str(root / relative_path))
        path = verify_file(binding, root=root)
        if path != (root / relative_path).resolve():
            raise GeneratorContractError('source binding key/path mismatch')
        bound.add(relative_path.as_posix())
    actual_python = {p.relative_to(root).as_posix() for p in root.rglob('*.py') if '.git' not in p.parts}
    if actual_python != {p for p in bound if p.endswith('.py')}:
        raise GeneratorContractError('source Python inventory mismatch (unbound or missing code)')
    if (root / '.git').exists():
        actual_revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if actual_revision != revision:
            raise GeneratorContractError('source git revision mismatch')
    else:
        if 'source_revision.json' not in bound:
            raise GeneratorContractError('archive source requires a bound source_revision.json')
        record = json.loads((root / 'source_revision.json').read_text())
        if record.get('revision') != revision:
            raise GeneratorContractError('archive source revision mismatch')
    return root


def load_tensor_state(checkpoint: Mapping[str, Any], key: str | None, *, metadata_namespace: bool = False) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = verify_file(checkpoint)
    # Official StyleSwin training saves its inert argparse.Namespace alongside
    # tensor states. Permit only that known standard-library metadata type;
    # arbitrary pickled classes/functions remain rejected by weights_only.
    allowlist = torch.serialization.safe_globals([argparse.Namespace]) if metadata_namespace else nullcontext()
    with allowlist:
        value = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(value, Mapping):
        raise GeneratorContractError('checkpoint is not a state mapping')
    top_level_keys = list(value.keys())
    if key is not None:
        if key not in value or not isinstance(value[key], Mapping):
            raise GeneratorContractError(f'explicit checkpoint state key absent: {key}')
        value = value[key]
    if not value or any(not isinstance(name, str) or not torch.is_tensor(tensor) for name, tensor in value.items()):
        raise GeneratorContractError('selected checkpoint state is not tensor-only')
    for name, tensor in value.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise GeneratorContractError(f'checkpoint contains nonfinite tensor: {name}')
    return dict(value), {'path': str(path), 'sha256': checkpoint['sha256'], 'state_key': key,
                         'top_level_keys': top_level_keys if key is not None else None,
                         'discriminator_state_present': 'd' in top_level_keys,
                         'weights_only': True,
                         'metadata_allowlist': ['argparse.Namespace'] if metadata_namespace else []}


@contextmanager
def _temporary_modules(replacements: Mapping[str, types.ModuleType]):
    missing = object()
    previous = {key: sys.modules.get(key, missing) for key in replacements}
    try:
        sys.modules.update(replacements)
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _module(path: Path, name: str, *, package: bool = False):
    locations = [str(path.parent)] if package else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=locations)
    if spec is None or spec.loader is None:
        raise GeneratorContractError('cannot create source import specification')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def fused_leaky_relu(input: torch.Tensor, bias: torch.Tensor | None = None,
                     negative_slope: float = 0.2, scale: float = math.sqrt(2)) -> torch.Tensor:
    """Native StyleGAN/StyleSwin CPU formula, also differentiable on CUDA."""
    if input.ndim < 2:
        raise GeneratorContractError('fused activation expects a channel dimension')
    if bias is not None:
        if bias.ndim != 1 or bias.numel() != input.shape[1]:
            raise GeneratorContractError('fused activation bias/channel mismatch')
        input = input + bias.view(1, -1, *([1] * (input.ndim - 2)))
    return F.leaky_relu(input, negative_slope=negative_slope) * scale


def upfirdn2d(input: torch.Tensor, kernel: torch.Tensor, up: int = 1,
              down: int = 1, pad: tuple[int, int] = (0, 0)) -> torch.Tensor:
    """Exact zero-insert/pad/FIR/decimate native reference, without JIT CUDA.

    Kernel coefficients, factor scaling (done by official Upsample), asymmetric
    padding, flip, and phase are unchanged. It is intentionally not bilinear
    interpolation. The official CUDA extension is an acceleration of this rule.
    """
    if input.ndim != 4 or kernel.ndim != 2:
        raise GeneratorContractError('upfirdn2d requires NCHW and a 2D kernel')
    if not isinstance(up, int) or not isinstance(down, int) or up < 1 or down < 1:
        raise GeneratorContractError('up/down must be positive integer factors')
    if len(pad) != 2 or any(not isinstance(value, int) for value in pad):
        raise GeneratorContractError('native upfirdn2d pad requires two integers')
    # Preserve the official CPU reference's view/permutation order, including
    # backward accumulation order. Source: Microsoft StyleSwin op/upfirdn2d.py
    # (MIT licensed); no fused extension is loaded or compiled.
    n, channel, height, width = input.shape
    out = input.reshape(-1, height, width, 1)
    out = out.view(-1, height, 1, width, 1, 1)
    out = F.pad(out, (0, 0, 0, up - 1, 0, 0, 0, up - 1))
    out = out.view(-1, height * up, width * up, 1)
    pad0, pad1 = pad
    out = F.pad(out, (0, 0, max(pad0, 0), max(pad1, 0), max(pad0, 0), max(pad1, 0)))
    out = out[:, max(-pad0, 0):out.shape[1] - max(-pad1, 0),
              max(-pad0, 0):out.shape[2] - max(-pad1, 0), :]
    out = out.permute(0, 3, 1, 2)
    padded_h, padded_w = height * up + pad0 + pad1, width * up + pad0 + pad1
    out = out.reshape(-1, 1, padded_h, padded_w)
    out = F.conv2d(out, kernel.to(device=input.device, dtype=input.dtype).flip((0, 1)).view(1, 1, *kernel.shape))
    filtered_h, filtered_w = padded_h - kernel.shape[0] + 1, padded_w - kernel.shape[1] + 1
    out = out.reshape(-1, 1, filtered_h, filtered_w).permute(0, 2, 3, 1)
    out = out[:, ::down, ::down, :]
    out_h = (padded_h - kernel.shape[0]) // down + 1
    out_w = (padded_w - kernel.shape[1]) // down + 1
    return out.reshape(n, channel, out_h, out_w)


def _styleswin_module(root: Path, revision: str):
    prefix = '_lerra_styleswin_' + revision
    package = types.ModuleType(prefix)
    package.__path__ = [str(root / 'models')]
    sys.modules[prefix] = package
    ops = types.ModuleType('_lerra_styleswin_reference_ops')
    ops.fused_leaky_relu = fused_leaky_relu
    ops.upfirdn2d = upfirdn2d
    # Official generator has absolute "models.basic_layers" / "op" imports.
    # Scope those names to construction so SiT's models.py cannot collide.
    with _temporary_modules({'models': package, 'op': ops}):
        basic = _module(root / 'models/basic_layers.py', prefix + '.basic_layers')
        with _temporary_modules({'models.basic_layers': basic}):
            return _module(root / 'models/generator.py', prefix + '.generator')


def _strict_load(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    expected = model.state_dict()
    if set(expected) != set(state):
        missing = sorted(set(expected) - set(state))[:5]
        extra = sorted(set(state) - set(expected))[:5]
        raise GeneratorContractError(f'state keys differ: missing={missing}, unexpected={extra}')
    mismatches = [name for name in expected if expected[name].shape != state[name].shape]
    if mismatches:
        raise GeneratorContractError(f'state tensor shapes differ: {mismatches[:5]}')
    model.load_state_dict(state, strict=True)


def _require_fixed_args(args: Mapping[str, Any], fixed: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(args) - set(fixed)
    if unknown:
        raise GeneratorContractError(f'unsupported constructor arguments: {sorted(unknown)}')
    if any(args.get(key, value) != value for key, value in fixed.items()):
        raise GeneratorContractError('model constructor arguments change the pinned architecture')
    return dict(fixed)


@dataclass
class GeneratorBackend:
    kind: str
    model: nn.Module
    spec: dict[str, Any]
    provenance: dict[str, Any]
    task: Any = None
    sampler: Any = None
    decoder: nn.Module | None = None
    _source_root: Path | None = field(default=None, repr=False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def make_inputs(self, *, seed: int, batch_size: int = 1,
                    labels: list[int] | None = None) -> dict[str, torch.Tensor]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise GeneratorContractError('batch size must be positive')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        if self.kind == 'styleswin_256':
            if labels is not None:
                raise GeneratorContractError('the pinned StyleSwin is unconditional')
            return {'noise': torch.randn(batch_size, 512, generator=generator).to(self.device)}
        if self.kind == 'sit_b_2':
            num_classes, allowed, null_label = conditioning(self.spec)
            if labels is None:
                indices = torch.randint(0, len(allowed), (batch_size,), generator=generator)
                y = torch.tensor(allowed, dtype=torch.long)[indices]
            else:
                if len(labels) != batch_size or any(isinstance(v, bool) or not isinstance(v, int) or v not in allowed for v in labels):
                    raise GeneratorContractError('SiT labels must match the declared conditioning and batch size')
                y = torch.tensor(labels, dtype=torch.long)
            return {'x': torch.randn(batch_size, 4, 32, 32, generator=generator).to(self.device),
                    't': torch.rand(batch_size, generator=generator).to(self.device), 'y': y.to(self.device)}
        if self.kind == 'guided_diffusion':
            if labels is not None:
                raise GeneratorContractError('the pinned U-Net is unconditional')
            return {'x': torch.randn(batch_size, 3, 256, 256, generator=generator).to(self.device),
                    'timesteps': torch.randint(0, 1000, (batch_size,), generator=generator).to(self.device)}
        raise GeneratorContractError('unknown generator kind')

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        output = self.model(**inputs)
        if self.kind == 'styleswin_256':
            output = output[0]
        return output

    def readout(self):
        from .generative_readout import guided_readout, sit_readout, styleswin_readout
        return {'guided_diffusion': guided_readout, 'sit_b_2': sit_readout,
                'styleswin_256': styleswin_readout}[self.kind](self.model)

    def load_decoder(self) -> nn.Module:
        if self.kind != 'sit_b_2':
            raise GeneratorContractError('only the latent SiT backend uses a VAE decoder')
        if self.decoder is None:
            binding = self.spec.get('vae')
            if not isinstance(binding, Mapping):
                raise GeneratorContractError('RGB SiT generation requires a bound local VAE')
            path = Path(binding['path']).resolve()
            files = binding.get('files')
            if not files:
                raise GeneratorContractError('VAE requires bound config and safetensor files')
            if isinstance(files, Mapping):
                records = []
                for relative, item in files.items():
                    record = {'path': str(path / relative), 'sha256': item} if isinstance(item, str) else dict(item)
                    record.setdefault('path', str(path / relative))
                    records.append(record)
            else:
                records = files
            verified = {verify_file(record, root=path).relative_to(path).as_posix() for record in records}
            if 'config.json' not in verified or 'diffusion_pytorch_model.safetensors' not in verified:
                raise GeneratorContractError('VAE config and safetensor file must both be verified')
            if any(p.name.endswith('.index.json') for p in path.iterdir()):
                raise GeneratorContractError('unreviewed VAE sharded/index override is not allowed')
            from diffusers.models import AutoencoderKL
            decoder, loading = AutoencoderKL.from_pretrained(
                str(path), local_files_only=True, use_safetensors=True,
                output_loading_info=True, low_cpu_mem_usage=False)
            if any(loading.get(name) for name in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                raise GeneratorContractError('VAE state did not load exactly; random/missing decoder weights are forbidden')
            if (float(decoder.config.scaling_factor) != 0.18215 or
                    int(decoder.config.latent_channels) != 4 or
                    int(decoder.config.out_channels) != 3):
                raise GeneratorContractError('VAE scaling/channels differ from official SiT sampling')
            self.decoder = decoder.to(self.device).eval()
            self.decoder.requires_grad_(False)
        return self.decoder

    @torch.no_grad()
    def sample_rgb(self, *, seed: int, batch_size: int = 1,
                   labels: list[int] | None = None, steps: int = 250,
                   cfg_scale: float = 4.0, sampling_method: str = 'dopri5',
                   atol: float = 1e-6, rtol: float = 1e-3) -> torch.Tensor:
        """Generate real 256px RGB; [0,1] clipping is only the export boundary.

        SiT uses the exact official transport and its first-three-channel CFG,
        then its public VAE decoder; a latent tensor is never labeled RGB.
        StyleSwin retains truncation=1 and the official ImageNet normalization.
        Guided diffusion retains the sampler schedule bound in model_args.
        """
        if self.model.training:
            raise GeneratorContractError('generation requires model.eval()')
        inputs = self.make_inputs(seed=seed, batch_size=batch_size, labels=labels)
        if self.kind == 'styleswin_256':
            raw = self.forward(inputs)
            mean = raw.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = raw.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            rgb = raw * std + mean
        elif self.kind == 'guided_diffusion':
            if self.sampler is None:
                raise GeneratorContractError('guided diffusion sampler is unavailable')
            # Native DDIM still draws unused per-step noise at eta=0. Isolate
            # those draws so paired comparisons preserve the caller's RNG.
            devices = [self.device.index] if self.device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    torch.cuda.default_generators[devices[0]].manual_seed(seed)
                raw = self.sampler.ddim_sample_loop(self.model, (batch_size, 3, 256, 256),
                                                    noise=inputs['x'], clip_denoised=True,
                                                    eta=0.0, device=self.device, progress=False)
            rgb = (raw + 1) / 2
        elif self.kind == 'sit_b_2':
            if not isinstance(steps, int) or steps < 2:
                raise GeneratorContractError('SiT requires at least two integration points')
            if not math.isfinite(cfg_scale) or cfg_scale < 0:
                raise GeneratorContractError('CFG scale must be finite and nonnegative')
            if sampling_method not in {'dopri5', 'dopri8', 'euler', 'midpoint', 'rk4', 'heun3', 'bosh3', 'fehlberg2', 'adaptive_heun'}:
                raise GeneratorContractError('unsupported official ODE integration method')
            if atol <= 0 or rtol <= 0 or not math.isfinite(atol + rtol):
                raise GeneratorContractError('ODE tolerances must be finite and positive')
            # Transport imports are lazy; revalidate source immediately before
            # importing it, even when the backend has been held in memory.
            verify_source(self.spec['source'])
            transport = _module(self._source_root / 'transport/__init__.py',
                                '_lerra_sit_transport_' + self.spec['source']['revision'], package=True)
            task = transport.create_transport('Linear', 'velocity', None, None, None)
            sample_fn = transport.Sampler(task).sample_ode(sampling_method=sampling_method,
                                                           num_steps=steps, atol=atol, rtol=rtol, reverse=False)
            z = torch.cat([inputs['x'], inputs['x']], dim=0)
            y = torch.cat([inputs['y'], torch.full_like(inputs['y'], conditioning(self.spec)[2])], dim=0)
            sampled = sample_fn(z, self.model.forward_with_cfg, y=y, cfg_scale=cfg_scale)[-1]
            latents = sampled.chunk(2, dim=0)[0]
            raw = self.load_decoder().decode(latents / 0.18215).sample
            rgb = (raw + 1) / 2
        else:
            raise GeneratorContractError('unknown generator kind')
        if tuple(rgb.shape) != (batch_size, 3, 256, 256) or not bool(torch.isfinite(rgb).all()):
            raise GeneratorContractError('native sampling did not produce finite 256px RGB')
        return rgb.clamp(0, 1)


def load_generator(spec: Mapping[str, Any], device: str | torch.device = 'cpu',
                   *, load_decoder: bool = False) -> GeneratorBackend:
    return _load_generator(spec, device, load_decoder=load_decoder)


def initialize_caltech_sit_embedding(state, *, classes, seed):
    """New Caltech labels have no semantic correspondence to ImageNet labels."""
    name='y_embedder.embedding_table.weight'
    old=state.get(name)
    if classes != 256 or not torch.is_tensor(old) or tuple(old.shape)!=(1001,768):
        raise GeneratorContractError('SiT transfer requires the pinned 1000-class embedding and 256 new Caltech classes')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    replacement=torch.randn((257,768),generator=generator,dtype=old.dtype)*.02
    replacement[256].copy_(old[1000].cpu())
    return {**state,name:replacement}


def load_reference_initialization(spec: Mapping[str, Any], recipe: Mapping[str, Any],
                                  device: str | torch.device = 'cpu') -> GeneratorBackend:
    """Explicit transfer-only entrypoint; never presents initialization as clean evidence."""
    if spec.get('dataset')!='caltech256' or spec.get('kind') not in {'guided_diffusion','sit_b_2'}:
        raise GeneratorContractError('native reference adaptation is restricted to Caltech U-Net/SiT')
    if recipe.get('development_seed')!=100:
        raise GeneratorContractError('reference initialization uses predeclared development seed 100')
    return _load_generator(spec,device,load_decoder=spec['kind']=='sit_b_2',initialization=recipe)


def _load_generator(spec: Mapping[str, Any], device: str | torch.device = 'cpu',
                    *, load_decoder: bool = False, initialization=None) -> GeneratorBackend:
    """Load strict pinned public states; no checkpoint means an error, not init."""
    spec = dict(spec)
    kind = spec.get('kind')
    if initialization is None:
        require_reference_contract(spec)
    else:
        spec['checkpoint']=dict(spec['initialization_checkpoint'])
        spec['checkpoint_key']=spec.get('initialization_checkpoint_key',spec.get('checkpoint_key'))
    if kind not in {'guided_diffusion', 'sit_b_2', 'styleswin_256'}:
        raise GeneratorContractError('unsupported generator kind')
    source = verify_source(spec['source'])
    if 'checkpoint_key' not in spec:
        raise GeneratorContractError('checkpoint_key must explicitly select raw/EMA state')
    key = spec['checkpoint_key']
    if kind == 'styleswin_256' and key != 'g_ema':
        raise GeneratorContractError('StyleSwin public evaluation uses explicit g_ema state')
    metadata_namespace = spec.get('checkpoint_metadata_namespace', False)
    if not isinstance(metadata_namespace, bool) or (metadata_namespace and kind != 'styleswin_256'):
        raise GeneratorContractError('Namespace metadata opt-in is only supported for an explicit StyleSwin training checkpoint')
    state, checkpoint_provenance = load_tensor_state(spec['checkpoint'], key, metadata_namespace=metadata_namespace)
    if initialization is not None and kind=='sit_b_2':
        if initialization.get('conditioning_initialization')!='new_caltech_embeddings_preserve_imagenet_null':
            raise GeneratorContractError('explicit Caltech conditioning initialization is required')
        state=initialize_caltech_sit_embedding(state,classes=conditioning(spec)[0],seed=initialization['development_seed'])
    task = sampler = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)  # construction RNG is discarded after strict loading
        if kind == 'sit_b_2':
            args = _require_fixed_args(spec.get('model_args', {}),
                                       {'input_size': 32, 'num_classes': conditioning(spec)[0], 'learn_sigma': True})
            module = _module(source / 'models.py', '_lerra_sit_' + spec['source']['revision'])
            model = module.SiT_models['SiT-B/2'](**args)
            # Explicit non-fused attention follows the original attention order;
            # it also avoids device-specific fused-kernel selection on new timm.
            for submodule in model.modules():
                if hasattr(submodule, 'fused_attn'):
                    submodule.fused_attn = False
            if tuple(model.final_layer.linear.weight.shape) != (32, 768):
                raise GeneratorContractError('published SiT-B/2 output must be [32,768] with native variance-channel discard')
            output_semantics = 'latent_velocity_4x32x32_native_variance_discard_then_frozen_vae_to_rgb'
            dimension = 768
        elif kind == 'styleswin_256':
            args = _require_fixed_args(spec.get('model_args', {}),
                                       {'size': 256, 'style_dim': 512, 'n_mlp': 8,
                                        'channel_multiplier': 2, 'lr_mlp': 0.01,
                                        'enable_full_resolution': 8, 'mlp_ratio': 4,
                                        'use_checkpoint': False, 'qkv_bias': True,
                                        'qk_scale': None, 'drop_rate': 0, 'attn_drop_rate': 0})
            module = _styleswin_module(source, spec['source']['revision'])
            model = module.Generator(**args)
            shapes = [tuple(layer.conv.weight.shape) for layer in model.to_rgbs]
            if shapes != [(3, d, 1, 1) for d in (512, 512, 512, 512, 256, 128, 128)]:
                raise GeneratorContractError('StyleSwin joint ToRGB layout differs from d=2560 contract')
            dimension = 2560
            output_semantics = 'rgb_imagenet_mean_std_normalized'
        else:
            if key is not None:
                raise GeneratorContractError('the existing public U-Net checkpoint is a raw state dict')
            args = dict(spec.get('model_args', {}))
            required = {'image_size': 256, 'class_cond': False, 'learn_sigma': True,
                        'num_channels': 256, 'num_res_blocks': 2, 'resblock_updown': True,
                        'use_scale_shift_norm': True, 'num_head_channels': 64,
                        'use_kl': False, 'predict_xstart': False, 'use_fp16': False,
                        'diffusion_steps': 1000}
            if any(args.get(name) != value for name, value in required.items()):
                raise GeneratorContractError('U-Net constructor differs from existing FP32 RGB protocol')
            for name, module in tuple(sys.modules.items()):
                if name == 'guided_diffusion' or name.startswith('guided_diffusion.'):
                    filename = getattr(module, '__file__', None)
                    if filename and not Path(filename).resolve().is_relative_to(source):
                        raise GeneratorContractError('guided_diffusion was imported from a different source')
            sys.path.insert(0, str(source))
            try:
                util = importlib.import_module('guided_diffusion.script_util')
                model, sampler = util.create_model_and_diffusion(**args)
                task = util.create_gaussian_diffusion(steps=args['diffusion_steps'], learn_sigma=True,
                                                      noise_schedule=args['noise_schedule'], use_kl=False,
                                                      predict_xstart=False,
                                                      rescale_timesteps=args['rescale_timesteps'],
                                                      rescale_learned_sigmas=args['rescale_learned_sigmas'],
                                                      timestep_respacing='')
            finally:
                sys.path.remove(str(source))
            if tuple(model.out[2].weight.shape) != (6, 256, 3, 3):
                raise GeneratorContractError('U-Net output must be [6,256,3,3]')
            dimension = 2304
            output_semantics = 'rgb_noise_prediction_first3_plus_learned_variance_last3'
        _strict_load(model, state)
    del state
    model = model.to(device).eval()
    # Analytic insertion modifies selected physical parameters, not gradients.
    # A future native FT executor must explicitly unfreeze its selected model.
    model.requires_grad_(False)
    provenance = {'kind': kind, 'pretrained': True, 'training_updates': 0,
                  'source_revision': spec['source']['revision'], 'checkpoint': checkpoint_provenance,
                  'readout_dimension': dimension, 'output_semantics': output_semantics,
                  'dtype': 'float32', 'styleswin_ops': 'native_reference_pytorch_no_jit' if kind == 'styleswin_256' else None,
                  'sit_attention': 'explicit_nonfused' if kind == 'sit_b_2' else None}
    if initialization is not None:
        provenance['reference_initialization_only']=True
        provenance['conditioning_initialization']=initialization.get('conditioning_initialization')
    result = GeneratorBackend(kind, model, spec, provenance, task=task, sampler=sampler, _source_root=source)
    if load_decoder:
        result.load_decoder()
    return result
