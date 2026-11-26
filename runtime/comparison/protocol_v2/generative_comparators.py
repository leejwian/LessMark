"""Native watermark cores adapted to fixed pretrained generator backbones.

``build_comparator(backend, name, settings, seed, bank=...)`` returns loss,
truth-free recovery, direct insertion, selected-parameter snapshots and exact
base-bound delta persistence. Learning policy belongs to the training runner.
The four learned methods use genuine internal tensors, never a fabricated
concatenated LERRA readout. TATTOOED keeps native RNG and last-write semantics,
but gathers only selected positions instead of copying a full generator.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn

from . import generative_readout as gr
from .methods import METHODS, Method, Recovery, decode, make_assets as native_assets, uchida_vector
from .storage import state_hash


LEARNED = ('uchida', 'greedy', 'neuralmark', 'deepmarks')
DEFAULT_INTERNAL = {
    'guided_diffusion': 'output_blocks.17.0.out_layers.3.weight',
    'sit_b_2': 'blocks.11.mlp.fc2.weight',
    'styleswin_256': 'layers.6.blocks.1.mlp.fc2.weight',
}
_AFFINE = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _payload(value):
    code = torch.as_tensor(value).detach().cpu()
    if code.shape != (256,) or not bool(((code == 0) | (code == 1)).all()):
        raise ValueError('comparison requires exactly 256 binary payload coordinates')
    return code.to(torch.uint8).numpy()


def _checkpoint_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('verified lowercase checkpoint SHA256 required')
    return value


def default_layers(backend, name):
    if name == 'lerra':
        return [b.module_path + '.weight' for b in backend.readout().branches]
    if name in LEARNED:
        if backend.kind not in DEFAULT_INTERNAL:
            raise ValueError('unknown generator has no declared native internal layer')
        return [DEFAULT_INTERNAL[backend.kind]]
    if name != 'tattooed':
        raise ValueError('unsupported native comparator')
    terminal = set(default_layers(backend, 'lerra'))
    selected = set()
    for path, module in backend.model.named_modules():
        # Official StyleSwin EqualLinear has the same actual dense weight role.
        if isinstance(module, _AFFINE) or (type(module).__name__ == 'EqualLinear' and
                                           hasattr(module, 'weight') and module.weight.ndim == 2):
            selected.add(path + '.weight')
    # Preserve actual named-parameter order; never sort it after key creation.
    return [name for name, _ in backend.model.named_parameters() if name in selected and name not in terminal]


def default_settings(backend, name):
    common = {'layers': default_layers(backend, name), 'tolerance': 1e-8}
    if name == 'lerra':
        return {**common, 'alpha': .02, 'carrier': 'Low'}
    if name == 'greedy':
        return {**common, 'width': 253, 'divider': 2, 'margin': .1}
    if name == 'neuralmark':
        return {**common, 'projection_std': 1.}
    if name == 'deepmarks':
        return {**common, 'reduction': 'mean'}
    if name == 'tattooed':
        return {**common, 'ratio': .00125, 'gamma': .0009, 'maxiter': 1000,
                'ldpc_seed': 8, 'parameter_seed': 1, 'spreading_seed': 2}
    return common


def _hash_state_except(model, selected):
    return {name: gr._tensor_hash(tensor) for name, tensor in model.state_dict().items() if name not in selected}


class Comparator:
    def __init__(self, backend, name, settings, seed, *, bank=None):
        if name not in METHODS or type(seed) is not int or seed < 0:
            raise ValueError('unknown native method or invalid fixed key seed')
        self.backend, self.model, self.name, self.seed = backend, backend.model, name, seed
        self.settings = {**default_settings(backend, name), **copy.deepcopy(settings)}
        if not isinstance(self.settings['tolerance'], (int, float)) or not math.isfinite(self.settings['tolerance']) or self.settings['tolerance'] <= 0:
            raise ValueError('finite positive native validation tolerance required')
        self._settings_hash = state_hash(self.settings)
        self.layers = tuple(self.settings['layers'])
        parameters = dict(self.model.named_parameters())
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError('selected native weight tensors must be nonempty and distinct')
        for name in self.layers:
            if name not in parameters or not name.endswith('.weight'):
                raise ValueError('missing actual selected weight tensor: ' + name)
            if parameters[name].dtype != torch.float32 or not bool(torch.isfinite(parameters[name]).all()):
                raise ValueError('comparison requires finite FP32 master weights')
            if not parameters[name].is_contiguous():
                raise ValueError('selected generator master weights must be contiguous')
        self._parameters = tuple((name, parameters[name]) for name in self.layers)
        if len({id(p) for _, p in self._parameters}) != len(self._parameters):
            raise ValueError('aliased selected parameters are unsupported')
        self._base_hashes = {name: gr._tensor_hash(p) for name, p in self._parameters}
        self._unchanged_hashes = _hash_state_except(self.model, set(self.layers))
        self.readout = None
        if self.name == 'lerra':
            self.readout = backend.readout()
            if self.layers != tuple(b.module_path + '.weight' for b in self.readout.branches):
                raise ValueError('LERRA layers must be the actual declared affine readout')
            if bank is None or 'bases' not in bank:
                raise ValueError('LERRA requires a completed carrier bank')
            basis = bank['bases']['256/' + self.settings['carrier']]
            q = torch.ones(self.readout.outputs, dtype=torch.float64)/math.sqrt(self.readout.outputs)
            self.assets = gr.make_assets(self.readout, basis, q, alpha=self.settings['alpha'],
                                         tolerance=self.settings['tolerance'])
            for key in ('cutoff', 'tie_atol'):
                if key in self.settings:
                    value = self.settings[key]
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or (key == 'tie_atol' and value < 0):
                        raise ValueError('invalid fixed LERRA attribution setting: ' + key)
                    self.assets[key] = value
            self.native = None
        else:
            if self.name not in ('greedy', 'tattooed') and len(self.layers) != 1:
                raise ValueError('native comparator requires one whole weight tensor')
            self._validate_capacity()
            self.assets = native_assets(self.name, self.model, self.settings, seed, bits=256)
            if self.name in ('uchida', 'neuralmark') and int(torch.linalg.matrix_rank(self.assets['P'].double())) != 256:
                raise ValueError('fixed native projection is rank deficient; no key resampling')
            self.native = Method(self.name, self.assets)
        self._assets_hash = state_hash(self.assets)
        self._sparse = None
        if self.name == 'tattooed':
            positions = np.asarray(self.assets['positions'], dtype=np.int64)
            unique, reverse_first = np.unique(positions[::-1], return_index=True)
            last = len(positions)-1-reverse_first
            self._sparse = {'unique': unique, 'last': last,
                            'inverse': np.searchsorted(unique, positions)}
            self._base_sparse = self._gather_sparse()
            self._base = None
        else:
            self._base = {name: p.detach().cpu().clone() for name, p in self._parameters}

    def _validate_capacity(self):
        if self.name == 'uchida' and uchida_vector(self._parameters[0][1]).numel() < 256:
            raise ValueError('Uchida actual reduced dimension is smaller than 256')
        if self.name == 'greedy':
            width = self.settings['width']
            divider = self.settings['divider']
            if type(width) is not int or width < 1 or not math.isfinite(divider) or divider < 0:
                raise ValueError('invalid native Greedy pooling settings')
            if divider and int(width/divider+.5) >= width:
                raise ValueError('native Greedy settings erase the entire payload signal')
            if not math.isfinite(self.settings['margin']) or self.settings['margin'] <= 0:
                raise ValueError('native Greedy margin must be finite and positive')
            if any(p.numel()//width*width < 256*width for _, p in self._parameters):
                raise ValueError('Greedy cannot upsample a small tensor into artificial capacity')
        if self.name == 'neuralmark' and (not math.isfinite(self.settings['projection_std']) or self.settings['projection_std'] <= 0):
            raise ValueError('NeuralMark projection standard deviation must be finite and positive')
        if self.name == 'deepmarks' and self._parameters[0][1][0].numel() < 256:
            raise ValueError('DeepMarks requires actual reduced dimension >=256')
        if self.name == 'tattooed':
            for key in ('ldpc_seed', 'parameter_seed', 'spreading_seed', 'maxiter'):
                if type(self.settings[key]) is not int or self.settings[key] < (1 if key == 'maxiter' else 0):
                    raise ValueError('invalid fixed TATTOOED key/decoder setting')
            if not math.isfinite(self.settings['ratio']) or not 0 < self.settings['ratio'] <= 1:
                raise ValueError('invalid TATTOOED ratio')
            if not math.isfinite(self.settings['gamma']) or self.settings['gamma'] <= 0:
                raise ValueError('invalid TATTOOED gain')

    def target_parameters(self):
        return self._parameters

    def configure_trainable(self):
        if self.name not in LEARNED:
            raise ValueError('direct insertion has no optimization parameters')
        self.model.requires_grad_(False)
        for _, parameter in self._parameters:
            parameter.requires_grad_(True)
        return self._parameters

    def provenance(self):
        return {'schema': 'native_generator_comparator_v1', 'method': self.name,
                'generator_kind': self.backend.kind, 'payload_bits': 256, 'key_seed': self.seed,
                'settings': copy.deepcopy(self.settings), 'layers': list(self.layers),
                'shapes': {name: list(p.shape) for name, p in self._parameters},
                'assets_sha256': self._assets_hash, 'base_selected_sha256': dict(self._base_hashes),
                'selection': 'actual_internal_whole_weight_tensor' if self.name in LEARNED else
                             ('actual_affine_readout' if self.name == 'lerra' else 'ordered_declared_generator_weights'),
                'loss_and_decoder': 'unchanged_native_core',
                'task_training_policy': 'caller_declared_pretrained_layer_only_adaptation',
                'truth_available_to_decoder': False,
                'readout': self.readout.contract() if self.readout else None}

    def _assert_assets(self):
        if state_hash(self.assets) != self._assets_hash or state_hash(self.settings) != self._settings_hash:
            raise ValueError('native key/assets changed after comparator construction')

    def loss(self, payload):
        if self.name not in LEARNED:
            raise ValueError('direct method has no optimizer watermark loss')
        return self.native.loss(self.model, _payload(payload))

    def recover(self, collusion=False):
        if self.name == 'lerra':
            signal = gr.extract(self.readout, self.assets)
            return decode(signal/self.assets['alpha'], 2/3) if collusion else decode(signal, erasure_at_zero=True)
        if self.name == 'tattooed':
            return self._tattooed_recover()
        return self.native.recover(self.model, collusion=collusion)

    @torch.no_grad()
    def insert(self, payload):
        code = _payload(payload)
        self._assert_assets()
        if self.name == 'lerra':
            gr.insert(self.readout, code, self.assets)
        elif self.name == 'tattooed':
            from .tattooed import encode_message
            values = self._gather_sparse()
            rng = np.random.RandomState(self.assets['spreading_seed'])
            for bit in encode_message(code, self.assets):
                # Consume the complete original spread vector, including repeat
                # positions. Only the final occurrence at each index survives.
                spread = rng.choice([-1, 1], size=len(self.assets['positions']))
                values = (values + self.assets['gamma']*spread[self._sparse['last']]*bit).astype(np.float32)
            self._scatter_sparse(values)
        else:
            raise ValueError('native learning-based insertion requires task+watermark training')
        return self.recover()

    def _sparse_parts(self):
        offset = 0
        unique = self._sparse['unique']
        for name, parameter in self._parameters:
            left, right = np.searchsorted(unique, [offset, offset + parameter.numel()])
            yield name, parameter, left, right, unique[left:right]-offset
            offset += parameter.numel()

    def _gather_sparse(self):
        parts = []
        for _, parameter, left, right, indices in self._sparse_parts():
            if left != right:
                selected = parameter.detach().reshape(-1)[torch.as_tensor(indices, device=parameter.device)]
                parts.append(selected.cpu().numpy().copy())
        return np.concatenate(parts)

    @torch.no_grad()
    def _scatter_sparse(self, values):
        if values.shape != self._sparse['unique'].shape or not np.isfinite(values).all():
            raise ValueError('invalid sparse native weight values')
        for _, parameter, left, right, indices in self._sparse_parts():
            if left != right:
                parameter.reshape(-1)[torch.as_tensor(indices, device=parameter.device)] = torch.from_numpy(values[left:right]).to(parameter)

    def _tattooed_recover(self):
        # Exact frozen tattooed.extract projection and LDPC decode, after sparse
        # gathering replaces only its memory-heavy flatten-and-index operation.
        from pyldpc import decode as ldpc_decode, get_message
        a = self.assets
        w = self._gather_sparse()[self._sparse['inverse']]
        if not np.isfinite(w).all():
            return Recovery(None, 'invalid', details={'reason': 'nonfinite weights'})
        rng = np.random.RandomState(a['spreading_seed'])
        y = np.array([rng.choice([-1, 1], size=len(w)) @ w for _ in range(a['encoded_length'])])
        gain = float(np.mean(y[:200]*a['preamble']))
        if not np.isfinite(gain) or gain == 0:
            return Recovery(None, 'decode_failure', y, {'reason': 'undefined gain'})
        sigma = float(np.std(y[:200]*a['preamble']/gain))
        if not np.isfinite(sigma) or sigma <= 0:
            return Recovery(None, 'decode_failure', y, {'reason': 'undefined SNR'})
        snr = float(-20*np.log10(sigma))
        bits, converged = [], True
        with warnings.catch_warnings(record=True) as caught:
            for j in range(200, len(y), 384):
                decoded = ldpc_decode(a['H'], y[j:j+384]/gain, snr, maxiter=a['maxiter'])
                converged &= bool(np.all(a['H'] @ decoded % 2 == 0))
                bits.extend(get_message(a['G'], decoded))
        return Recovery(np.asarray(bits[:a['bits']], dtype=np.uint8),
                        'ok' if converged else 'decode_nonconvergence', y,
                        {'gain': gain, 'snr': snr, 'parity_valid': converged,
                         'warnings': len(caught), 'encoded_length': len(y), 'message_length': a['bits']})

    def snapshot(self):
        if self.name == 'tattooed':
            return {'schema': 'native_comparator_sparse_snapshot_v1', 'layers': self.layers,
                    'values': self._gather_sparse(),
                    'selected_sha256': {name: gr._tensor_hash(p) for name, p in self._parameters}}
        return {name: parameter.detach().cpu().clone() for name, parameter in self._parameters}

    @torch.no_grad()
    def restore(self, snapshot):
        if snapshot.get('schema') == 'native_comparator_sparse_snapshot_v1':
            if self.name != 'tattooed' or tuple(snapshot['layers']) != self.layers:
                raise ValueError('sparse snapshot comparator mismatch')
            self._scatter_sparse(snapshot['values'])
            if any(gr._tensor_hash(p) != snapshot['selected_sha256'][name] for name, p in self._parameters):
                raise ValueError('weights outside sparse watermark positions changed; reload public base')
            return
        if set(snapshot) != set(self.layers):
            raise ValueError('snapshot selected layer set changed')
        for name, parameter in self._parameters:
            value = snapshot[name]
            if value.shape != parameter.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('invalid selected-parameter snapshot')
        for name, parameter in self._parameters:
            parameter.copy_(snapshot[name].to(parameter))

    def assert_unselected_unchanged(self):
        if _hash_state_except(self.model, set(self.layers)) != self._unchanged_hashes:
            raise ValueError('unselected generator parameter or buffer changed')
        if self.readout is not None:
            for (name, parameter), rows in zip(self._parameters, self.readout.rows):
                unselected = sorted(set(range(parameter.shape[0])) - set(rows))
                if unselected and not torch.equal(parameter.detach()[unselected].cpu(), self._base[name][unselected]):
                    raise ValueError('unselected native output rows changed')


def build_comparator(backend, name, settings=None, seed=0, *, bank=None):
    return Comparator(backend, name, settings or {}, seed, bank=bank)


def _file_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def save_delta(comparator, path, *, base_checkpoint_sha256):
    """Persist exact changed values, never an entire generator checkpoint."""
    _checkpoint_hash(base_checkpoint_sha256)
    comparator._assert_assets()
    comparator.assert_unselected_unchanged()
    tensors = {}
    results = {}
    sparse_parts = {name: (left, right, indices) for name, _, left, right, indices in comparator._sparse_parts()} if comparator.name == 'tattooed' else None
    for name, parameter in comparator.target_parameters():
        current = parameter.detach().cpu()
        if not bool(torch.isfinite(current).all()):
            raise ValueError('cannot persist nonfinite selected generator weights')
        results[name] = gr._tensor_hash(current)
        if sparse_parts is not None:
            left, right, indices = sparse_parts[name]
            at = torch.as_tensor(indices, dtype=torch.long)
            values = current.reshape(-1)[at]
            original = torch.from_numpy(comparator._base_sparse[left:right])
            changed = values != original
            at, values = at[changed], values[changed]
            # Prove no caller changed an unrecorded coordinate in this tensor.
            check = current.clone()
            check.reshape(-1)[torch.as_tensor(indices, dtype=torch.long)] = original
            if gr._tensor_hash(check) != comparator._base_hashes[name]:
                raise ValueError('TATTOOED weights outside declared sparse positions changed')
            del check
        else:
            changed = current.reshape(-1) != comparator._base[name].reshape(-1)
            at = torch.nonzero(changed, as_tuple=False).reshape(-1)
            values = current.reshape(-1)[at]
        sparse = sparse_parts is not None or at.numel()*12 < current.numel()*4
        tensors[name] = {'shape': list(current.shape), 'dtype': 'torch.float32',
                         'encoding': 'sparse_values' if sparse else 'selected_dense_values',
                         'indices': at.clone() if sparse else torch.empty(0, dtype=torch.long),
                         'values': values.clone() if sparse else current.clone()}
    payload = {'schema': 'native_generator_comparator_delta_v1',
               'comparator': comparator.provenance(), 'base_checkpoint_sha256': base_checkpoint_sha256,
               'base_selected_sha256': comparator._base_hashes,
               'result_selected_sha256': results,
               'unchanged_state_sha256': comparator._unchanged_hashes, 'tensors': tensors}
    gr._atomic_save(payload, path)
    return {**_file_record(path), 'schema': payload['schema'],
            'changed_tensors': sum(payload['base_selected_sha256'][name] != results[name] for name in results),
            'stored_values': sum(item['values'].numel() for item in tensors.values()),
            'full_model_saved': False, 'comparator': comparator.provenance()}


@torch.no_grad()
def apply_delta(comparator, path, *, base_checkpoint_sha256):
    _checkpoint_hash(base_checkpoint_sha256)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'native_generator_comparator_delta_v1' or payload.get('base_checkpoint_sha256') != base_checkpoint_sha256 or payload.get('comparator') != comparator.provenance():
        raise ValueError('delta comparator/public checkpoint provenance changed')
    comparator._assert_assets()
    comparator.assert_unselected_unchanged()
    if payload.get('unchanged_state_sha256') != comparator._unchanged_hashes or payload.get('base_selected_sha256') != comparator._base_hashes or set(payload['tensors']) != set(comparator.layers):
        raise ValueError('delta base/unchanged state or tensor coverage mismatch')
    current = {name: gr._tensor_hash(p) for name, p in comparator.target_parameters()}
    if current == payload['result_selected_sha256']:
        return {'applied': False, 'already_applied': True, 'delta': _file_record(path)}
    if current != payload['base_selected_sha256']:
        raise ValueError('delta cannot be applied over changed selected base weights')
    # Validate every tensor before changing any model parameter.
    for name, parameter in comparator.target_parameters():
        item = payload['tensors'][name]
        if item.get('shape') != list(parameter.shape) or item.get('dtype') != 'torch.float32':
            raise ValueError('delta tensor metadata mismatch')
        values, indices = item['values'], item['indices']
        if values.dtype != torch.float32 or not bool(torch.isfinite(values).all()) or indices.dtype != torch.long:
            raise ValueError('invalid delta tensor values')
        if item['encoding'] == 'sparse_values':
            if indices.ndim != 1 or values.shape != indices.shape or (len(indices) and
               (indices[0] < 0 or indices[-1] >= parameter.numel() or not bool((indices[1:] > indices[:-1]).all()))):
                raise ValueError('invalid delta sparse coordinates')
        elif item['encoding'] != 'selected_dense_values' or values.shape != parameter.shape or indices.numel():
            raise ValueError('invalid dense selected-tensor encoding')
        candidate = parameter.detach().cpu().clone()
        if item['encoding'] == 'sparse_values':
            candidate.reshape(-1)[indices] = values
        else:
            candidate.copy_(values)
        if gr._tensor_hash(candidate) != payload['result_selected_sha256'][name]:
            raise ValueError('delta reconstruction hash mismatch')
        del candidate
    for name, parameter in comparator.target_parameters():
        item = payload['tensors'][name]
        if item['encoding'] == 'sparse_values':
            parameter.reshape(-1)[item['indices'].to(parameter.device)] = item['values'].to(parameter)
        else:
            parameter.copy_(item['values'].to(parameter))
    return {'applied': True, 'already_applied': False, 'delta': _file_record(path)}
