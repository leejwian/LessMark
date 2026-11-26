"""Project a registered FFHQ training photo onto a frozen StyleSwin generator."""

import math

import numpy as np
import torch
from PIL import Image

from runtime.comparison.protocol_v2.generative_embed import selected_rows


@torch.enable_grad()
def collect_styleswin_photo_features(backend, photo_path, photo_index, projection, positions_per_forward):
    """Return CPU features, flat pixel positions, best post-update z and initial/best post-update RGB MSE.

    Pass a frozen clean StyleSwin backend, the aligned PNG from the verified
    manifest, its zero-based row, the design's projection settings, and the
    configured position count. Apply projection["runtime"] in the launching
    process before loading the CUDA backend. The caller can add
    result["features"] directly to the readout's streaming second moment.
    """
    if backend.kind != "styleswin_256" or backend.model.training:
        raise ValueError("expected the clean StyleSwin-256 generator in evaluation mode")
    if any(parameter.requires_grad for parameter in backend.model.parameters()):
        raise ValueError("freeze the clean generator before projecting images")
    if photo_index < 0 or projection["steps"] < 1:
        raise ValueError("expected a nonnegative manifest row and a positive projection budget")

    with Image.open(photo_path) as image:
        if image.format != "PNG" or image.width != image.height:
            raise ValueError("expected a square aligned FFHQ PNG")
        resized = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        target = torch.from_numpy(np.array(resized)).permute(2, 0, 1).unsqueeze(0)

    z = backend.make_inputs(seed=projection["initial_seed_base"] + photo_index)["noise"]
    z = z.detach().requires_grad_(True)
    target = target.to(z).div_(255)
    mean = z.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = z.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    optimizer = torch.optim.Adam([z], lr=projection["learning_rate"], betas=tuple(projection["betas"]))
    steps = projection["steps"]
    best_mse = math.inf
    for step in range(steps + 1):
        # Step zero supplies the baseline MSE and the gradient for the first update.
        with torch.set_grad_enabled(step < steps):
            raw = backend.forward({"noise": z})
            loss = (raw * std + mean - target).square().mean()
        mse = loss.item()
        if not math.isfinite(mse):
            raise ValueError(f"non-finite RGB projection loss at row {photo_index}, step {step}")
        if step == 0:
            initial_mse = mse
        elif mse < best_mse:
            best_mse = mse
            best_update = step
            best_z = z.detach().clone()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    readout = backend.readout()
    with torch.no_grad():
        with readout.capture() as capture:
            backend.forward({"noise": best_z})
        positions = selected_rows(capture.count, positions_per_forward, projection["position_seed_base"] + photo_index)
        features = capture.feature_rows(positions).cpu()
    return {
        "features": features,
        "positions": positions,
        "z": best_z.cpu(),
        "initial_mse": initial_mse,
        "best_update": best_update,
        "best_mse": best_mse,
    }
