"""Small CPU model and watermark implementations shared by campaign tests."""
from collections import Counter
import random

import numpy as np
import torch

from experiment_2.code import run_strength_selection as campaign
from runtime.comparison.protocol_v2.methods import Recovery


class TinyBackend:
    def __init__(self):
        self.model = torch.nn.Linear(2, 1, bias=False)
        self.source = torch.tensor([[0.25, -0.125]])
        self.steps = []
        self.calls = Counter()
        self.reset_training()

    def reset_training(self):
        self.model.train()
        with torch.no_grad():
            self.model.weight.copy_(self.source)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9)

    def train_step(self, step, watermark_loss=None):
        self.model.train()
        self.optimizer.zero_grad()
        # A resumed endpoint requires restoration of model, momentum and all three RNGs.
        inputs = torch.tensor([[1.0, random.random() + np.random.random()]]) + torch.rand(1, 2)
        loss = (self.model(inputs) - 0.75).square().mean()
        if watermark_loss is not None:
            loss = loss + watermark_loss()
        loss.backward()
        self.optimizer.step()
        self.steps.append((step, watermark_loss is not None))

    def training_state_dict(self):
        return self.optimizer.state_dict()

    def load_training_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def prepare_quality(self, directory):
        self.calls['clean_quality'] += 1
        campaign.write_json(directory / 'clean-count.json', {'correct': 3})

    def quality(self, directory):
        self.calls['marked_quality'] += 1
        clean_count = campaign.read_json(directory / 'clean-count.json')['correct']
        return {'examples': 4, 'clean_correct': clean_count, 'marked_correct': 2,
                'clean_top1_pct': 75.0, 'marked_top1_pct': 50.0, 'accuracy_drop_pp': 25.0}


class TinyWatermark:
    def __init__(self, backend, method):
        self.backend = backend
        self.method = method

    def loss(self):
        self.backend.calls['loss'] += 1
        return self.backend.model.weight.square().sum()

    @torch.no_grad()
    def insert(self, strength):
        self.backend.calls['direct_insert'] += 1
        self.backend.model.weight[0, 0] += strength

    @torch.no_grad()
    def recover(self):
        self.backend.calls[self.method + '_recover'] += 1
        return Recovery(np.repeat(self.backend.model.weight.numpy() > 0, 128).astype(np.uint8), 'ok')


def prepare_assets(design, condition, method, backend, *, cache_dir):
    backend.calls[method + '_assets'] += 1
    return {'reference_weight': backend.model.weight.detach().cpu().clone()}


def bind_watermark(design, condition, method, backend, assets):
    return TinyWatermark(backend, method)
