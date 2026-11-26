"""Matched native source training and seeded terminal RGB validation for experiment 2."""

import fcntl
import hashlib
import inspect
import io
import sys
import types
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, __version__ as pillow_version

from experiment_2.code.input_paths import read_registered_json
from runtime.comparison.protocol_v2 import generative_models as models
from runtime.comparison.protocol_v2.generative_quality import rgb_ssim, rgb_tensor
from runtime.comparison.protocol_v2.generative_reference import _seed, native_loss
from runtime.comparison.protocol_v2.rgb import center_crop
from runtime.comparison.protocol_v2.storage import atomic_path, freeze_json, identity, read_json, state_hash, write_json


CONFIG = Path(__file__).resolve().parents[1] / 'config'


def _model_spec(design, condition):
    binding = design['strength']['historical_centers_source']
    path = (CONFIG / binding['path_relative_to_this_file']).resolve()
    records = read_registered_json(models.verify_file({'path': str(path), 'sha256': binding['sha256']}))
    row, = (r for r in records['rows'] if r['condition'] == condition and r['method'] == 'lerra')
    spec = read_registered_json(models.verify_file(row['contract']))['model']
    expected = design['embedding']['fixed_source_checkpoints'][condition]
    if spec['checkpoint'] != expected['checkpoint'] or spec['checkpoint_key'] != expected.get('checkpoint_key'):
        raise ValueError('generator C0 or state key differs from the experiment 2 registration')
    if spec['kind'] == 'styleswin_256':
        spec['checkpoint_metadata_namespace'] = True
    return spec


class _RealTrainingPhotos:
    def __init__(self, design, condition, spec):
        training = design['embedding']['native_generator_training']
        domain = training['conditions'][condition]['train_pool']
        pool = training['train_pools'][domain]
        self.domain = domain
        self.checked_files = set()
        self.ffhq_rgb_cache = None
        self.ffhq_source_stamps = {}
        self.labels = None
        self.classes = None
        if domain == 'ffhq':
            first, last = pool['official_training_ids_inclusive']
            base = Path(pool['image_directory'])
            self.rows = [{'path': str(base / f'{i:05d}.png')} for i in range(first, last + 1)]
        else:
            self.rows = read_registered_json(models.verify_file(pool['manifest']))
            if len({r['class_id'] for r in self.rows}) != pool['represented_classes']:
                raise ValueError('source training class inventory differs')
            if spec['kind'] == 'sit_b_2':
                if domain == 'imagenet':
                    catalog = design['embedding']['lessmark_real_image_calibration']['photo_lists']['imagenet']
                    ids = sorted({r['class_id'] for r in read_registered_json(models.verify_file(catalog))})
                else:
                    ids = spec['conditioning']['class_ids']
                if len(set(ids)) != pool['source_class_count'] or len(ids) != spec['conditioning']['num_classes']:
                    raise ValueError('incomplete source conditioning vocabulary')
                indices = {name: i for i, name in enumerate(ids)}
                self.classes = list(ids)
                self.labels = [indices[r['class_id']] for r in self.rows]
        if len(self.rows) != pool['available_images'] or len({r['path'] for r in self.rows}) != len(self.rows):
            raise ValueError('source training pool membership differs')
        self.order_key = None
        self.order = None

    def batch_indices(self, step, batch, seed):
        result = []
        while len(result) < batch:
            epoch, offset = divmod(step * batch + len(result), len(self.rows))
            if self.order_key != (seed, epoch):
                gen = torch.Generator().manual_seed(_seed(seed, 'epoch', epoch))
                self.order = torch.randperm(len(self.rows), generator=gen)
                self.order_key = (seed, epoch)
            result.extend(self.order[offset:offset + batch - len(result)].tolist())
        return result

    def register_ffhq_rgb_cache(self, manifest):
        self.ffhq_rgb_cache = Path(manifest).parent / 'ffhq_training_rgb256'
        freeze_json(self.ffhq_rgb_cache / 'origin.json', {
            'schema': 'ffhq_native_training_rgb_v1', 'training_inputs_sha256': models._sha256(Path(manifest)),
            'pillow': pillow_version, 'transform': 'convert RGB, resize 256x256 PIL BILINEAR',
            'storage': 'uint8 HWC, before normalization',
        })

    def _ffhq_pixels(self, index, source):
        def resize(stream):
            with Image.open(stream) as image:
                return torch.from_numpy(np.array(image.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)))

        if self.ffhq_rgb_cache is None:
            return resize(source)
        cache = self.ffhq_rgb_cache / f'{index // 1000:03d}' / f'{index:05d}.pt'
        def source_stamp():
            stat = source.stat()
            return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

        stamp = source_stamp()
        if cache.exists():
            record = torch.load(cache, map_location='cpu', weights_only=True)
            if record['source'] != str(source):
                raise ValueError(f'FFHQ cached source differs: {source}')
            if self.ffhq_source_stamps.get(index, record['source_stamp']) != stamp:
                if models._sha256(source) != record['source_sha256']:
                    raise ValueError(f'FFHQ source bytes changed after first training use: {source}')
                self.ffhq_source_stamps[index] = stamp
            pixels = record['pixels']
            if (pixels.shape != (256, 256, 3) or pixels.dtype != torch.uint8 or
                    hashlib.sha256(pixels.numpy().tobytes()).hexdigest() != record['rgb_sha256']):
                raise ValueError(f'Damaged FFHQ training RGB: {cache}')
            return pixels
        raw = source.read_bytes()
        if source_stamp() != stamp:
            raise ValueError(f'FFHQ source changed while reading training input: {source}')
        pixels = resize(io.BytesIO(raw))
        record = {'source': str(source), 'source_stamp': stamp, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                  'pixels': pixels, 'rgb_sha256': hashlib.sha256(pixels.numpy().tobytes()).hexdigest()}
        with atomic_path(cache) as temporary, temporary.open('wb') as stream:
            torch.save(record, stream)
        return pixels

    def image(self, index, seed):
        row = self.rows[index]
        if self.domain == 'ffhq':
            pixels = self._ffhq_pixels(index, Path(row['path']))
            rgb = pixels.permute(2, 0, 1).float().div_(255)
            return (rgb - rgb.new_tensor([.485, .456, .406])[:, None, None]) / rgb.new_tensor([.229, .224, .225])[:, None, None]
        if index not in self.checked_files and 'sha256' in row:
            models.verify_file(row)
            self.checked_files.add(index)
        with Image.open(row['path']) as source:
            rgb = center_crop(source)
        return rgb.flip(-1) if _seed(seed, 'flip') % 2 else rgb


class _FusedLeakyReLU(torch.nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(channel))

    def forward(self, value):
        return models.fused_leaky_relu(value, self.bias)


def _wavelet_filter(value, kernel, up=1, down=1, pad=(0, 0)):
    # The source inverse Haar transform spells out the same x/y padding twice.
    if len(pad) == 4 and pad[:2] == pad[2:]:
        pad = pad[:2]
    return models.upfirdn2d(value, kernel, up, down, pad)


def _discriminator_source(native):
    prefix = '_lerra_styleswin_' + native.spec['source']['revision']
    root = native._source_root
    ops = types.ModuleType('_experiment2_styleswin_d_ops')
    ops.FusedLeakyReLU = _FusedLeakyReLU
    ops.upfirdn2d = _wavelet_filter
    modules = {'models': sys.modules[prefix], 'models.basic_layers': sys.modules[prefix + '.basic_layers'], 'op': ops}
    with models._temporary_modules(modules):
        discriminator = models._module(root / 'models/discriminator.py', prefix + '.discriminator')
    augment = models._module(root / 'utils/CRDiffAug.py', prefix + '.cr_diff_aug')
    return discriminator.Discriminator, augment.CR_DiffAug


@contextmanager
def _rng(seed, device):
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        yield


def _backward(loss):
    if loss.ndim or not torch.isfinite(loss.detach()).item():
        raise ValueError('native training produced an invalid scalar loss')
    loss.backward()


class GenerationBackend:
    def __init__(self, design, condition, device, recipe):
        self.condition = condition
        self.recipe = dict(recipe)
        if not 0 < recipe['microbatch_size'] <= recipe['batch_size'] or recipe['batch_size'] % recipe['microbatch_size']:
            raise ValueError('microbatches must partition the source batch')
        spec = _model_spec(design, condition)
        self.photos = _RealTrainingPhotos(design, condition, spec)
        self.native = models.load_generator(spec, device=device)
        self.flow_cache = None
        if self.native.kind == 'sit_b_2':
            # Construct the frozen VAE outside each arm's seeded dropout/noise stream.
            self.native.load_decoder()
        self.model = self.native.model
        # A CUDA parameter carries the resolved cuda:N even for the CLI's bare "cuda".
        self.device = self.native.device
        self.c0 = {k: v.detach().to('cpu', copy=True) for k, v in self.model.state_dict().items()}
        self.quality_config = design['generation_evaluation']
        self.terminal_updates, = design['checkpoint_schedule']['generation_updates']
        self.completed_updates = 0
        self.is_gan = self.native.kind == 'styleswin_256'
        if self.is_gan:
            self.d_class, self.augment = _discriminator_source(self.native)
        self.optimizer = self.discriminator = self.d_optimizer = None

    def prepare_training_manifest(self, path):
        """Commit one source schedule for this condition, shared across every arm."""
        indices = [self.photos.batch_indices(step, self.recipe['batch_size'], self.recipe['seed'])
                   for step in range(self.terminal_updates)]
        unique = len({index for batch in indices for index in batch})
        presentations = self.terminal_updates * self.recipe['batch_size']
        freeze_json(path, {
            'condition': self.condition, 'seed': self.recipe['seed'], 'recipe': self.recipe,
            'source_pool': self.photos.domain, 'source_rows': self.photos.rows,
            'source_class_vocabulary': self.photos.classes, 'source_labels': self.photos.labels,
            'order': 'zero-based source_rows indices, in presentation order for each consecutive update',
            'batch_indices_by_update': indices, 'unique_photos': unique, 'presentations': presentations,
            'repeat_presentations': presentations - unique,
            'augmentation_and_task_rng': 'generative_reference._seed(seed, "source_training", step, within_batch_index)',
            'native_model_rng': 'generative_reference._seed(seed, "model_rng", step)',
        })
        if self.photos.domain == 'ffhq':
            self.photos.register_ffhq_rgb_cache(path)
        if self.native.kind == 'sit_b_2':
            self.flow_cache = Path(path).parent / 'native_flow_inputs'
            self.flow_cache.mkdir(parents=True, exist_ok=True)
            freeze_json(self.flow_cache / 'origin.json', {
                'schema': 'source_linear_velocity_fp32_v1', 'condition': self.condition, 'recipe': self.recipe,
                'c0': self.native.spec['checkpoint']['sha256'], 'vae': self.native.spec['vae'],
                'training_inputs_sha256': models._sha256(Path(path)),
            })

    def _enable_generator_training(self):
        self.model.train().requires_grad_(True)
        if self.native.kind == 'sit_b_2':
            self.model.pos_embed.requires_grad_(False)

    def _start_optimizers(self):
        cfg = self.recipe
        self._enable_generator_training()
        if self.is_gan:
            with _rng(_seed(cfg['seed'], 'initial_discriminator'), self.device):
                self.discriminator = self.d_class(256, channel_multiplier=2, sn=True).to(self.device)
            def adam(parameters, lr, interval):
                ratio = interval / (1 + interval)
                return torch.optim.Adam(parameters, lr=lr * ratio, betas=tuple(b ** ratio for b in cfg['betas']),
                                        eps=cfg['eps'], foreach=False)
            self.optimizer = adam(self.model.parameters(), cfg['lr'], cfg['g_reg_every'])
            self.d_optimizer = adam(self.discriminator.parameters(), cfg['d_lr'], cfg['r1_every'])
        else:
            self.optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad),
                                               lr=cfg['lr'], betas=tuple(cfg['betas']), eps=cfg['eps'],
                                               weight_decay=cfg['weight_decay'], foreach=False)

    def reset_training(self):
        self.optimizer = self.d_optimizer = self.discriminator = None
        self.completed_updates = 0
        self.model.zero_grad(set_to_none=True)
        self.model.load_state_dict(self.c0, strict=True)
        self._start_optimizers()

    def training_state_dict(self):
        if self.optimizer is None:
            raise RuntimeError('source training not initialized')
        state = {'optimizer': self.optimizer.state_dict(), 'generator_updates': self.completed_updates}
        if self.is_gan:
            interval = self.recipe['r1_every']
            regularizations = (self.completed_updates + interval - 1) // interval
            state.update(discriminator=self.discriminator.state_dict(), d_optimizer=self.d_optimizer.state_dict(),
                         discriminator_ordinary_updates=self.completed_updates, discriminator_r1_updates=regularizations,
                         discriminator_updates=self.completed_updates + regularizations)
        return state

    def load_training_state_dict(self, aux):
        if self.optimizer is None:
            self._start_optimizers()
        self.completed_updates = aux['generator_updates']
        if type(self.completed_updates) != int or not 0 <= self.completed_updates <= self.terminal_updates:
            raise ValueError('invalid generator resume step')
        self.optimizer.load_state_dict(aux['optimizer'])
        if self.is_gan:
            regularizations = (self.completed_updates + self.recipe['r1_every'] - 1) // self.recipe['r1_every']
            if (aux['discriminator_ordinary_updates'], aux['discriminator_r1_updates'], aux['discriminator_updates']) != (
                    self.completed_updates, regularizations, self.completed_updates + regularizations):
                raise ValueError('StyleSwin resume discriminator step counts disagree with generator')
            self.discriminator.load_state_dict(aux['discriminator'], strict=True)
            self.d_optimizer.load_state_dict(aux['d_optimizer'])

    def _gan_phase(self, real_batches, step):
        r, d = self.recipe, self.discriminator
        d.train().requires_grad_(True)
        self.d_optimizer.zero_grad(set_to_none=True)
        count = len(real_batches)
        for real in real_batches:
            with torch.no_grad():
                fake = self.model(torch.randn(len(real), 512, device=self.device))[0]
            fp = d(fake)
            rp = d(real)
            real_aug, fake_aug = self.augment(real), self.augment(fake)
            fp_aug, rp_aug = d(fake_aug), d(real_aug)
            loss = F.softplus(fp).mean() + F.softplus(-rp).mean()
            loss = loss + r['bcr_fake'] * F.mse_loss(fp_aug, fp) + r['bcr_real'] * F.mse_loss(rp_aug, rp)
            _backward(loss / count)
        torch.nn.utils.clip_grad_norm_(d.parameters(), r['d_clip'], error_if_nonfinite=True)
        self.d_optimizer.step()
        if step % r['r1_every'] == 0:
            self.d_optimizer.zero_grad(set_to_none=True)
            for real in real_batches:
                image = real.detach().requires_grad_(True)
                pred = d(image)
                grad, = torch.autograd.grad(pred.sum(), image, create_graph=True)
                r1 = grad.square().flatten(1).sum(1).mean()
                _backward((r1 * (r['r1_gamma'] * .5 * r['r1_every']) + pred.sum() * 0) / count)
            self.d_optimizer.step()
        self.d_optimizer.zero_grad(set_to_none=True)
        d.requires_grad_(False)

    @torch.no_grad()
    def _flow_batch(self, step, indices, seeds):
        """Cache only frozen native_loss inputs and targets, before the trainable SiT call."""
        file = self.flow_cache / f'{step:04d}.pt' if self.flow_cache is not None else None
        if file is not None and file.exists():
            record = torch.load(file, map_location='cpu', weights_only=True)
            if record['step'] != step or record['indices'] != indices:
                raise ValueError('cached SiT source batch differs from the registered update')
            return tuple(t.to(self.device) for t in record['batch'])
        micro = self.recipe['microbatch_size']
        inputs, timesteps, targets = [], [], []
        for start in range(0, len(indices), micro):
            ids = indices[start:start + micro]
            draw_seeds = seeds[start:start + micro]
            images = torch.stack([self.photos.image(i, s) for i, s in zip(ids, draw_seeds)]).to(self.device)
            posterior = self.native.decoder.encode(images).latent_dist
            generators = [torch.Generator().manual_seed(s) for s in draw_seeds]
            shape = posterior.mean.shape[1:]
            eta = torch.stack([torch.randn(shape, generator=g) for g in generators]).to(self.device)
            latent = (posterior.mean + posterior.std * eta) * .18215
            noise = torch.stack([torch.randn(shape, generator=g) for g in generators]).to(self.device)
            t = torch.tensor([torch.rand((), generator=g).item() for g in generators], device=self.device)
            inputs.append((1 - t[:, None, None, None]) * noise + t[:, None, None, None] * latent)
            timesteps.append(t)
            targets.append(latent - noise)
        labels = torch.tensor([self.photos.labels[i] for i in indices], device=self.device, dtype=torch.long)
        result = (torch.cat(inputs), torch.cat(timesteps), torch.cat(targets), labels)
        if file is not None:
            with atomic_path(file) as tmp, tmp.open('wb') as stream:
                torch.save({'step': step, 'indices': indices, 'batch': [t.cpu() for t in result]}, stream)
        return result

    def train_step(self, step, watermark_loss=None):
        if step != self.completed_updates or not 0 <= step < self.terminal_updates:
            raise ValueError('generator update does not continue its registered training schedule')
        if self.optimizer is None:
            self._start_optimizers()
        self._enable_generator_training()
        cfg = self.recipe
        batch, micro = cfg['batch_size'], cfg['microbatch_size']
        indices = self.photos.batch_indices(step, batch, cfg['seed'])
        seeds = [_seed(cfg['seed'], 'source_training', step, j) for j in range(batch)]
        flow = None
        if self.native.kind == 'sit_b_2':
            flow = self._flow_batch(step, indices, seeds)
            chunks = flow[0].split(micro)
        else:
            # R1 and ordinary D use the same real minibatch in this update.
            images = torch.stack([self.photos.image(i, s) for i, s in zip(indices, seeds)]).to(self.device)
            chunks = images.split(micro)
        self.optimizer.zero_grad(set_to_none=True)
        with _rng(_seed(cfg['seed'], 'model_rng', step), self.device):
            if self.is_gan:
                self._gan_phase(chunks, step)
            task_total = 0.
            for j, chunk in enumerate(chunks):
                loc = slice(j * micro, (j + 1) * micro)
                if self.is_gan:
                    noise = torch.randn(micro, 512, device=self.device)
                    task_loss = F.softplus(-self.discriminator(self.model(noise)[0])).mean()
                elif flow is not None:
                    _, t, target, labels = flow
                    prediction = self.model(chunk, t[loc], labels[loc])
                    if prediction.shape != target[loc].shape:
                        raise ValueError('SiT native velocity shape differs from latent target')
                    task_loss = (prediction - target[loc]).square().flatten(1).mean(1).mean()
                else:
                    task_loss = native_loss(self.native, chunk, None, seeds[loc]).mean()
                task_loss = task_loss / len(chunks)
                _backward(task_loss)
                task_total += task_loss.detach().item()
            if watermark_loss is not None:
                _backward(watermark_loss())
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.completed_updates += 1
        return task_total

    def _quality_identity(self):
        start, end = self.quality_config['seed_ranges_inclusive']['validation']
        if end - start + 1 != self.quality_config['pairs_per_split']:
            raise ValueError('validation seed range/count mismatch')
        if self.native.kind == 'guided_diffusion':
            args = self.native.spec['model_args']
            sampler = {'kind': 'DDIM', 'eta': 0., 'clip_denoised': True, 'timesteps': args['diffusion_steps'],
                       'noise_schedule': args['noise_schedule'], 'timestep_respacing': args['timestep_respacing']}
        elif self.native.kind == 'sit_b_2':
            sampler = {'kind': 'native_velocity_ODE', 'path': 'Linear', 'sampling_method': 'dopri5', 'steps': 250,
                       'cfg_scale': 4., 'atol': 1e-6, 'rtol': 1e-3, 'vae_scale': .18215,
                       'null_label': self.native.spec['conditioning']['unconditional_label']}
        else:
            sampler = {'kind': 'native_StyleSwin', 'latent': 'standard_normal_512', 'truncation': 1.,
                       'RGB_mean': [.485, .456, .406], 'RGB_std': [.229, .224, .225]}
        return {'condition': self.condition, 'c0_sha256': self.native.spec['checkpoint']['sha256'],
                'seed_first': start, 'seed_last': end, 'sampler': sampler,
                'rgb': 'RGB 256x256 float32 [0,1]', 'pair_count': end-start+1}

    def _validation_inputs(self, settings):
        examples = []
        for seed in range(settings['seed_first'], settings['seed_last'] + 1):
            row = {'seed': seed}
            if self.native.kind == 'sit_b_2':
                # Native make_inputs consumes this CPU label draw BEFORE the Gaussian latent draw.
                allowed = self.native.spec['conditioning']['allowed_labels']
                rng = torch.Generator().manual_seed(seed)
                label = allowed[torch.randint(len(allowed), (1,), generator=rng).item()]
                row.update(label=label, source_class_id=self.photos.classes[label])
            examples.append(row)
        return examples

    def _validation_rgb(self, seed):
        """Distinguish per-seed undefined RGB from a failed or misconfigured sampler."""
        try:
            rgb = self.native.sample_rgb(seed=seed)
        except (models.GeneratorContractError, AssertionError) as exc:
            tb = exc.__traceback__
            while tb.tb_next is not None:
                tb = tb.tb_next
            frame = tb.tb_frame
            # The hash-pinned historical exporter uses one error for two checks.
            # Inspect its actual RGB only on that failure path to preserve shape errors.
            if (isinstance(exc, models.GeneratorContractError)
                    and frame.f_code is inspect.unwrap(models.GeneratorBackend.sample_rgb).__code__):
                value = frame.f_locals.get('rgb')
                if (isinstance(value, torch.Tensor) and value.shape == (1, 3, 256, 256)
                        and not torch.isfinite(value).all().item()):
                    return {'seed': seed, 'reason': 'nonfinite_rgb'}
            if (self.native.kind == 'sit_b_2' and isinstance(exc, AssertionError)
                    and frame.f_globals.get('__name__') == 'torchdiffeq._impl.rk_common'
                    and frame.f_code.co_name == '_adaptive_step'):
                dt = frame.f_locals.get('dt')
                if isinstance(dt, torch.Tensor) and dt.numel() == 1 and str(exc) == f'underflow in dt {dt.item()}':
                    return {'seed': seed, 'reason': 'ode_step_underflow'}
            raise
        if not isinstance(rgb, torch.Tensor) or rgb.shape != (1, 3, 256, 256):
            raise ValueError('validation sampler returned an invalid RGB shape')
        if not torch.isfinite(rgb).all().item():
            return {'seed': seed, 'reason': 'nonfinite_rgb'}
        return rgb

    def prepare_quality(self, cache_dir):
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / 'clean.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            settings = self._quality_identity()
            inputs = self._validation_inputs(settings)
            freeze_json(path / 'validation_inputs.json', inputs)
            cache_identity = dict(settings, clean_state_sha256=state_hash(self.model.state_dict()),
                                  validation_inputs_sha256=models._sha256(path / 'validation_inputs.json'))
            manifest = path / 'clean.json'
            freeze_json(manifest, cache_identity)
            self.model.eval()
            for row in inputs:
                seed = row['seed']
                file = path / f'{seed}.pt'
                if not file.exists():
                    sample = self._validation_rgb(seed)
                    with atomic_path(file) as tmp, tmp.open('wb') as stream:
                        torch.save(sample.cpu() if torch.is_tensor(sample) else sample, stream)
            write_json(path / 'ready.json', cache_identity)

    def quality(self, cache_dir, *, progress_dir=None):
        path = Path(cache_dir)
        clean_id = read_json(path / 'ready.json')
        settings = self._quality_identity()
        inputs = read_json(path / 'validation_inputs.json')
        if (clean_id != read_json(path / 'clean.json') or any(clean_id.get(k) != v for k, v in settings.items())
                or models._sha256(path / 'validation_inputs.json') != clean_id['validation_inputs_sha256']):
            raise ValueError('paired RGB cache identity differs')
        current = state_hash(self.model.state_dict())
        progress_root = Path(progress_dir) if progress_dir is not None else path
        progress_root.mkdir(parents=True, exist_ok=True)
        score_key = f"{identity(clean_id)}-{current}"
        progress = progress_root / f'scores-{score_key}.json'
        self.model.eval()
        with (progress_root / f'scores-{score_key}.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            scores = read_json(progress) if progress.exists() else []
            count = settings['pair_count']
            if len(scores) > count:
                raise ValueError('invalid RGB quality progress')
            for index in range(len(scores), count):
                seed = inputs[index]['seed']
                clean = torch.load(path / f'{seed}.pt', map_location=self.device, weights_only=True)
                if isinstance(clean, dict):
                    if clean.get('seed') != seed or clean.get('reason') not in {'nonfinite_rgb', 'ode_step_underflow'}:
                        raise ValueError('invalid clean RGB cache record')
                    scores.append(dict(clean, arm='clean'))
                else:
                    clean = rgb_tensor(clean)
                    actual = self._validation_rgb(seed)
                    if isinstance(actual, dict):
                        scores.append(dict(actual, arm='watermarked'))
                    else:
                        mse = (actual.double() - clean.double()).square().mean().item()
                        scores.append([mse, rgb_ssim(clean, actual)])
                write_json(progress, scores)
        failures = Counter(f"{s['arm']}:{s['reason']}" for s in scores if isinstance(s, dict))
        if failures:
            return {'pairs': count, 'invalid_pairs': sum(failures.values()), 'invalid_pair_reasons': dict(failures),
                    'quality_status': 'undefined', 'mean_mse': None, 'mean_ssim': None}
        return {'pairs': count, 'quality_status': 'ok', 'invalid_pairs': 0, 'mean_mse': sum(s[0] for s in scores) / count,
                'mean_ssim': sum(s[1] for s in scores) / count}
