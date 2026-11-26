#!/usr/bin/env python3
"""Exercise the imports and a CUDA kernel needed on a new training host."""
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main():
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    mismatches = []
    for line in requirements.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = "missing"
        if actual.split("+", 1)[0] != expected:
            mismatches.append(f"{name}: expected {expected}, installed {actual}")
    if mismatches:
        raise SystemExit("\n".join(mismatches))

    import torch
    import torchvision
    from diffusers.models import AutoencoderKL
    from pyldpc import make_ldpc
    from timm.models.vision_transformer import Attention
    from torchdiffeq import odeint

    if not torch.cuda.is_available():
        raise SystemExit("No usable NVIDIA GPU is visible to PyTorch.")
    devices = []
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        # Creating a context can succeed even when the wheel lacks a kernel
        # for this GPU architecture. Exercise a small numerical operation.
        probe = torch.ones(1, device=device) + 1
        if probe.item() != 2:
            raise SystemExit(f"CUDA kernel check failed on {device}")
        properties = torch.cuda.get_device_properties(device)
        devices.append({"device": str(device), "name": properties.name, "bytes": properties.total_memory})
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                      "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                      "cudnn": torch.backends.cudnn.version(), "devices": devices}, indent=2))


if __name__ == "__main__":
    main()
