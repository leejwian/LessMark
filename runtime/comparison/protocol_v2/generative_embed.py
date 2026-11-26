"""Seeded position selection for generator feature observations."""
import torch



def selected_rows(count, take, seed):
    if not 1 <= take <= count:
        raise ValueError('feature observations per forward exceed the actual native output positions')
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed))[:take]
