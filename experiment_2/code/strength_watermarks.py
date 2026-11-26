"""Fixed-payload insertion losses, direct insertion and independent native readers."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from runtime.comparison.protocol_v2 import generative_readout as gr
from runtime.comparison.protocol_v2.generative_comparators import Comparator, default_settings
from runtime.comparison.protocol_v2.generative_conditions import CONDITIONS
from runtime.comparison.protocol_v2.generative_models import verify_file
from runtime.comparison.protocol_v2.methods import Method, decode, make_assets
from runtime.comparison.protocol_v2.tattooed import encode_message
from experiment_2.code.classifier_lessmark import (
    insert_classifier_lessmark, prepare_classifier_lessmark, recover_classifier_lessmark,
)
from experiment_2.code.strength_calibration import calibrate_generator, generator_readout

PROJECT = Path(__file__).resolve().parents[2]
FIELDS = ('layers', 'key_seed', 'projection_std', 'width', 'divider', 'margin', 'tolerance',
          'reduction', 'maxiter', 'ratio', 'ldpc_seed', 'parameter_seed', 'spreading_seed')


def _settings(design, condition, method, backend):
    if condition in CONDITIONS:
        key = 700100 + 10000 * list(CONDITIONS).index(condition)
        recipe = {**default_settings(backend.native, method), 'key_seed': key}
        if method == 'tattooed':
            recipe.update(parameter_seed=key, spreading_seed=key + 1)
    else:
        binding = design['robustness']['eligible_scope']['classifier_layer_lists'][condition]
        path = verify_file({**binding, 'path': str(PROJECT / binding['path_from_project_root'])})
        recipe = json.loads(path.read_text())['config']['methods'][method]
    return {k: recipe[k] for k in FIELDS if k in recipe}


def _bits(design):
    bits = torch.tensor([int(b) for b in design['payload']['physical_payloads_by_recipient']['292']])
    if bits.shape != (256,) or not torch.all((bits == 0) | (bits == 1)) or bits.sum() != 128:
        raise ValueError('expected balanced 256-bit physical payload for recipient 292')
    return bits


def prepare_assets(design, condition, method, backend, *, cache_dir=None):
    """Run once under the runner's spec/condition/method lock, on C_T for LessMark."""
    if method == 'lerra':
        backend.model.eval()
        if condition in CONDITIONS:
            return calibrate_generator(design, condition, backend.native, cache_dir=cache_dir)
        settings = design['embedding']['lessmark_classifier_backbones']
        return prepare_classifier_lessmark(backend.model, backend.train_eval_dataset,
                                           backend.train_labels, settings, condition, cache_dir=cache_dir)
    recipe = _settings(design, condition, method, backend)
    parameters = dict(backend.model.named_parameters())
    layers = recipe['layers']
    if not layers or len(layers) != len(set(layers)) or any(n not in parameters for n in layers):
        raise ValueError(f'invalid ordered layer list for {condition}/{method}')
    assets = make_assets(method, backend.model, recipe, recipe['key_seed'])
    if method == 'tattooed':
        n = sum(parameters[name].numel() for name in layers)
        spreading = design['embedding']['tattooed_spreading_length']
        registered = spreading['conditions'][condition]
        if (assets['ratio'] != spreading['ratio'] or registered['eligible_coordinates'] != n
                or len(assets['positions']) != registered['spreading_length']
                or registered['spreading_length'] != n // 800):
            raise ValueError('TATOOED spreading length differs from the registered reference ratio')
        assets['eligible_coordinates'] = n
        assets['spreading_length'] = len(assets['positions'])
        assets['encoded_target'] = encode_message(_bits(design).numpy(), assets)
        # Safe weights-only serialization, without NumPy reconstruction pickles.
        for k in ('H', 'G', 'preamble', 'positions', 'encoded_target'):
            assets[k] = torch.from_numpy(np.asarray(assets[k]))
    elif method in ('uchida', 'neuralmark'):
        if int(torch.linalg.matrix_rank(assets['P'].double())) != 256:
            raise ValueError('fixed projection is rank deficient')
        if method == 'neuralmark':
            # Resolve the four fixed SHAKE-mask passes once. Indexing only the final
            # survivors avoids allocating four full-size masks at every update.
            mask = assets['filter_bits'].bool()
            selected = torch.arange(parameters[layers[0]].numel())
            for _ in range(4):
                repeats = selected.numel() // mask.numel()
                selected = selected[:repeats * mask.numel()].reshape(repeats, -1)[:, mask].reshape(-1)
            pooled = assets['P'].shape[0]
            assets['selected_indices'] = selected[:selected.numel() // pooled * pooled].clone()
    return assets


class _NeuralMark(Method):
    def signal(self, model):
        a = self.assets
        weight = model.get_parameter(a['layers'][0]).reshape(-1)
        selected = weight.index_select(0, a['selected_indices'])
        return selected.reshape(-1, a['P'].shape[0]).mean(0) @ a['P']


class _Learned:
    def __init__(self, name, assets, model, bits):
        self.model = model
        weight = model.get_parameter(assets['layers'][0])
        self.bits = bits.to(weight)
        device_assets = dict(assets)
        for k in ('P', 'U', 'X'):
            if k in assets:
                device_assets[k] = assets[k].to(weight)
        if name == 'neuralmark':
            device_assets['selected_indices'] = assets['selected_indices'].to(weight.device)
            self.method = _NeuralMark(name, device_assets)
        else:
            self.method = Method(name, device_assets)

    def loss(self):
        return self.method.loss(self.model, self.bits)

    @torch.no_grad()
    def recover(self):
        return self.method.recover(self.model)


class _LessMark:
    def __init__(self, condition, backend, assets, bits):
        self.backend, self.assets, self.bits = backend, assets, bits
        self.readout = generator_readout(backend.native) if condition in CONDITIONS else None

    @torch.no_grad()
    def insert(self, strength):
        if self.readout is None:
            return insert_classifier_lessmark(self.backend.model, self.assets, self.bits, strength)
        if not torch.equal(self.readout.weight_matrix().cpu().double(), self.assets['W0']):
            raise ValueError('direct insertion must start from clean terminal carrier weights')
        signal = gr.insert(self.readout, self.bits, {**self.assets, 'alpha': strength})
        return decode(signal, erasure_at_zero=True)

    @torch.no_grad()
    def recover(self):
        if self.readout is None:
            return recover_classifier_lessmark(self.backend.model, self.assets)
        return decode(gr.extract(self.readout, self.assets), erasure_at_zero=True)


class _Tattooed:
    # Same physical sparse traversal and target-independent LDPC reader as the original port.
    _sparse_parts = Comparator._sparse_parts
    _gather_sparse = Comparator._gather_sparse
    _scatter_sparse = Comparator._scatter_sparse
    _tattooed_recover = Comparator._tattooed_recover

    def __init__(self, assets, model):
        self.encoded_target = assets['encoded_target'].numpy()
        self.assets = {k: (v.numpy() if torch.is_tensor(v) else v)
                       for k, v in assets.items() if k != 'encoded_target'}
        self._parameters = [(n, model.get_parameter(n)) for n in assets['layers']]
        for n, p in self._parameters:
            if list(p.shape) != assets['shapes'][n] or p.dtype != torch.float32 or not p.is_contiguous():
                raise ValueError(f'incompatible TATOOED weight {n}')
        positions = self.assets['positions']
        unique, reverse_first = np.unique(positions[::-1], return_index=True)
        self._sparse = {'unique': unique, 'last': len(positions) - 1 - reverse_first,
                        'inverse': np.searchsorted(unique, positions)}

    @torch.no_grad()
    def insert(self, strength):
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError('TATOOED gamma must be finite and positive')
        values = self._gather_sparse()
        rng = np.random.RandomState(self.assets['spreading_seed'])
        for symbol in self.encoded_target:
            spread = rng.choice([-1, 1], size=self.assets['spreading_length'])
            values = (values + strength * spread[self._sparse['last']] * symbol).astype(np.float32)
        self._scatter_sparse(values)
        return self.recover()

    @torch.no_grad()
    def recover(self):
        return self._tattooed_recover()


def bind_watermark(design, condition, method, backend, assets):
    """Bind cached CPU keys to a fresh model; loss() carries no candidate coefficient."""
    if method == 'tattooed':
        return _Tattooed(assets, backend.model)
    bits = _bits(design)
    if method == 'lerra':
        return _LessMark(condition, backend, assets, bits)
    return _Learned(method, assets, backend.model, bits)
