"""Compute one LessMark bank from the actual clean terminal generator and training photos."""
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.sit_internal_readout_20261110 import internal_carrier as sit
from experiment_2.code.input_paths import read_registered_json
from experiment_2.code.styleswin_photo_features import collect_styleswin_photo_features
from runtime.comparison.protocol_v2 import generative_readout as gr
from runtime.comparison.protocol_v2.carriers import carriers
from runtime.comparison.protocol_v2.generative_conditions import conditioning
from runtime.comparison.protocol_v2.generative_embed import selected_rows
from runtime.comparison.protocol_v2.generative_models import verify_file
from runtime.comparison.protocol_v2.rgb import center_crop

CONFIG = Path(__file__).resolve().parents[1] / 'config'


def generator_readout(native):
    if native.kind == 'sit_b_2':
        return gr.AffineReadout(native.model, [gr.ReadoutBranch(sit.LAYER)], kind='sit_internal_fc2')
    return native.readout()


def _photos(settings, domain, native):
    binding = settings['photo_lists'][domain]
    labels = {}
    if domain == 'ffhq':
        path = verify_file({**binding, 'path': str(CONFIG / binding['path_relative_to_this_file'])})
        with path.open(newline='') as f:
            entries = [{'id': row['image_id'], 'sha256': row['file_sha256'],
                        'path': str(Path(binding['image_directory']) / (row['image_id'] + '.png'))}
                       for row in csv.DictReader(f)]
    else:
        entries = read_registered_json(verify_file(binding))
        if native.kind == 'sit_b_2':
            train = binding.get('source_training_manifest')
            membership = read_registered_json(verify_file(train)) if train else entries
            classes = sorted({e['class_id'] for e in membership})
            _, allowed, _ = conditioning(native.spec)
            if len(classes) != len(allowed):
                raise ValueError('SiT class map does not cover its declared source domain')
            labels = dict(zip(classes, allowed))
    if len(entries) != settings['photos_per_domain'] or len({e['sha256'] for e in entries}) != len(entries):
        raise ValueError('invalid LessMark calibration photo count or duplicate bytes')
    return entries, labels


def _style_rows(native, entry, index, settings, take, cache):
    path = verify_file(entry)
    projection = settings['styleswin_projection']
    save_path = cache / 'projections' / f'{index:04}.pt' if cache is not None else None
    if save_path is not None and save_path.exists():
        record = torch.load(save_path, map_location='cpu', weights_only=True)
        if record['sha256'] != entry['sha256'] or record['photo_id'] != entry['id']:
            raise ValueError('cached projection does not match its registered FFHQ photo')
        with native.readout().capture() as capture:
            native.forward({'noise': record['z'].to(native.device)})
        features = capture.feature_rows(record['positions']).cpu()
    else:
        record = collect_styleswin_photo_features(native, path, index, projection, take)
        features = record.pop('features')
        record.update(photo_id=entry['id'], sha256=entry['sha256'])
        if save_path is not None:
            gr._atomic_save(record, save_path)
    return features, record


def _sit_rows(native, inputs, positions, q, final):
    captured = {k: [] for k in ('h', 'x', 'gate', 'scale')}
    handles = sit._hooks(native, captured)
    try:
        output = native.forward(inputs)
    finally:
        for handle in handles:
            handle.remove()
    if (any(len(v) != 1 for v in captured.values()) or output.shape != (1, 4, 32, 32)
            or not torch.isfinite(output).all()):
        raise ValueError('unexpected SiT native velocity forward or internal hooks')
    features = captured['h'][0][0, positions].double()
    gate = captured['gate'][0].chunk(6, dim=-1)[-1]
    scale = captured['scale'][0].chunk(2, dim=-1)[-1]
    jq = sit.final_velocity_jq(captured['x'][0][:, positions], gate, q, scale,
                               final.linear.weight, final.norm_final.eps)
    # Each selected h h^T receives ||J(gate*q)||². The denominator remains the token count.
    return features * torch.linalg.vector_norm(jq[0], dim=-1, keepdim=True)


def _photo_rows(native, readout, entry, index, draws, take, label_map, seq_rng, pos_rng, q, final):
    with Image.open(verify_file(entry)) as image:
        x = center_crop(image).unsqueeze(0).to(native.device)
    if final is not None:
        posterior = native.load_decoder().encode(x).latent_dist
        y = torch.tensor([label_map[entry['class_id']]], device=native.device)
    observations = []
    for r in range(draws):
        i = index * draws + r
        rng = seq_rng if seq_rng is not None else torch.Generator().manual_seed(100 + i)
        if final is not None:
            eta = torch.randn(posterior.mean.shape, generator=rng).to(x.device)
            latent = (posterior.mean + posterior.std * eta) * .18215
            noise = torch.randn(latent.shape, generator=rng).to(x.device)
            t = torch.rand(1, generator=rng).to(x.device)
            inputs = {'x': (1-t) * noise + t * latent, 't': t, 'y': y}
            positions = selected_rows(256, take, 100100+i)
            observations.append(_sit_rows(native, inputs, positions, q, final))
        else:
            t = torch.randint(native.task.num_timesteps, (1,), generator=rng, device=rng.device).to(x.device)
            noise = torch.randn(x.shape, generator=rng, device=rng.device).to(x.device)
            with readout.capture() as cap:
                native.forward({'x': native.task.q_sample(x, t, noise=noise), 'timesteps': t})
            positions = (torch.from_numpy(pos_rng.choice(cap.count, take, replace=False)) if pos_rng is not None
                         else selected_rows(cap.count, take, 100100+i))
            observations.append(cap.feature_rows(positions))
    return torch.cat(observations)


@torch.no_grad()
def calibrate_generator(design, condition, native, *, cache_dir=None):
    """cache_dir belongs exclusively to this recipe and C_T digest, assigned by the locked runner."""
    settings = design['embedding']['lessmark_real_image_calibration']
    spec = settings['conditions'][condition]
    entries, labels = _photos(settings, spec['photo_list'], native)
    take, draws = settings['positions_per_forward'], spec['feature_forwards_per_photo']
    readout = generator_readout(native)
    final = None
    if native.kind == 'sit_b_2':
        _, final = sit._architecture(native)
        q = sit.balanced_q(readout.outputs)
    else:
        q = torch.ones(readout.outputs, dtype=torch.float64) / math.sqrt(readout.outputs)
    device_q = q.to(native.device)
    is_style = native.kind == 'styleswin_256'
    if is_style and (not torch.are_deterministic_algorithms_enabled() or torch.backends.cudnn.benchmark
                     or (native.device.type == 'cuda' and os.getenv('CUBLAS_WORKSPACE_CONFIG') != ':4096:8')):
        raise ValueError('FFHQ projection requires the registered deterministic process configuration')
    # Only the ImageNet U-Net uses sequential, device-local Torch and NumPy PCG64 streams.
    seq_rng = torch.Generator(device=native.device).manual_seed(100) if condition == 'unet_imagenet' else None
    pos_rng = np.random.default_rng(100) if seq_rng is not None else None
    cache = Path(cache_dir) if cache_dir is not None else None
    state_path = cache / 'moment.pt' if cache is not None else None
    start, records = 0, []
    if state_path is not None and state_path.is_file():
        saved = torch.load(state_path, weights_only=True, map_location='cpu')
        if saved['condition'] != condition or saved['device_type'] != native.device.type:
            raise ValueError('incompatible LessMark moment checkpoint')
        start, records, total = saved['next_photo'], saved['projections'], saved['total'].to(native.device)
        if seq_rng is not None:
            seq_rng.set_state(saved['torch_rng'])
            pos_rng.bit_generator.state = saved['position_rng']
        del saved
    else:
        total = torch.zeros(readout.d, readout.d, dtype=torch.float64, device=native.device)
    if not 0 <= start <= len(entries) or len(records) != (start if is_style else 0):
        raise ValueError('invalid LessMark moment checkpoint progress')
    flags = [p.requires_grad for p in native.model.parameters()]
    try:
        native.model.requires_grad_(False)
        for j in range(start, len(entries)):
            if is_style:
                features, record = _style_rows(native, entries[j], j, settings, take, cache)
                records.append(record)
            else:
                features = _photo_rows(native, readout, entries[j], j, draws, take, labels,
                                       seq_rng, pos_rng, device_q, final)
            if tuple(features.shape) != (draws * take, readout.d) or not torch.isfinite(features).all():
                raise ValueError(f'invalid LessMark observations for photo {j}')
            rows = features.to(total)
            total.addmm_(rows.T, rows)
            if state_path is not None and ((j + 1) % 16 == 0 or j + 1 == len(entries)):
                state = {'condition': condition, 'next_photo': j + 1, 'total': total.cpu(),
                         'projections': records, 'device_type': native.device.type}
                if seq_rng is not None:
                    state.update(torch_rng=seq_rng.get_state(), position_rng=pos_rng.bit_generator.state)
                gr._atomic_save(state, state_path)
    finally:
        for p, flag in zip(native.model.parameters(), flags):
            p.requires_grad_(flag)
    count = len(entries) * draws * take
    if count != spec['feature_rows'] or count <= readout.d:
        raise ValueError('incomplete generator calibration sample')
    moment = total.div_(count).cpu()
    bank = sit.low_basis(moment) if final is not None else carriers(moment, [256], [])
    assets = gr.make_assets(readout, bank['bases']['256/Low'], q, alpha=1.)
    assets.update(condition=condition, feature_rows=count, photo_sha256=[e['sha256'] for e in entries])
    if is_style:
        assets['projections'] = records
    return assets
