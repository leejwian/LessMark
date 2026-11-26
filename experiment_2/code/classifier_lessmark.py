"""Calibrate, insert and read the experiment 2 LessMark classifier backbone weights."""

import math
from pathlib import Path

import torch

from runtime.comparison.protocol_v2.carriers import carriers, proportional_sample, q_vector
from runtime.comparison.protocol_v2.data import batches
from runtime.comparison.protocol_v2.generative_readout import AffineReadout, ReadoutBranch, _atomic_save
from runtime.comparison.protocol_v2.methods import decode, lerra_insert, lerra_signal


def _backbone_readout(model, weight_name, weight_shape):
    path, suffix = weight_name.rsplit(".", 1)
    module = model.get_submodule(path)
    if suffix != "weight" or not isinstance(module, torch.nn.Conv2d):
        raise ValueError(f"expected a classifier backbone convolution at {weight_name}")
    if tuple(module.weight.shape) != tuple(weight_shape):
        raise ValueError(f"unexpected convolution weight shape at {weight_name}")
    return AffineReadout(model, [ReadoutBranch(path)], kind="classifier_backbone_conv")


@torch.no_grad()
def _patch_moment(readout, loader, spatial_shape, *, total=None, image_count=0, checkpoint=None, indices=None):
    """Accumulate one uncentered FP64 outer product per spatial convolution patch."""
    device = readout.modules[0].weight.device
    if total is None:
        total = torch.zeros((readout.d, readout.d), device=device, dtype=torch.float64)
    else:
        total = total.to(device)
    row_count = image_count * math.prod(spatial_shape)
    for images, _ in loader:
        with readout.capture() as captured:
            readout.model(images.to(device))
        if captured.layout != (len(images), *spatial_shape):
            raise ValueError("classifier convolution output grid differs from the design")
        rows = captured.feature_rows().double()
        total.addmm_(rows.T, rows)
        image_count += len(images)
        row_count += len(rows)
        if checkpoint is not None and (image_count % 512 == 0 or image_count == len(indices)):
            _atomic_save({"indices": indices, "images": image_count, "total": total.cpu()}, checkpoint)
    if row_count < readout.d + 1:
        raise ValueError("insufficient classifier backbone features for the registered basis")
    return total.div_(row_count), image_count, row_count


@torch.no_grad()
def prepare_classifier_lessmark(clean_model, train_eval_dataset, train_labels, settings, condition,
                               *, batch_size=32, cache_dir=None):
    """Use the task-trained C_T in eval mode with the cleaned training set and its evaluation transform.

    Pass embedding.lessmark_classifier_backbones as settings. Keep the returned
    assets for all strengths and subsequent readouts, including after head replacement.
    Sample indices are positions in the supplied ordered training membership.
    """
    registered = settings["conditions"][condition]
    readout = _backbone_readout(clean_model, registered["weight_name"], registered["weight_shape"])
    convolution = readout.modules[0]
    if (convolution.padding_mode != "zeros" or convolution.padding != tuple(registered["padding"])
            or convolution.stride != (1, 1) or convolution.dilation != (1, 1)):
        raise ValueError("classifier convolution padding, stride or dilation differs from the design")
    count = settings["calibration_images"]
    bits = settings["payload_bits"]
    if any(module.training for module in clean_model.modules()):
        raise ValueError("calibrate the clean terminal model in evaluation mode")
    if len(train_labels) != len(train_eval_dataset) or not 0 < count <= len(train_labels):
        raise ValueError("expected matching training images and labels sufficient for the calibration sample")
    if [readout.outputs, readout.d] != registered["matrix_shape"] or 2 * bits > readout.d:
        raise ValueError("classifier carrier dimensions disagree with the design")
    spatial_shape = settings["spatial_shape"]
    if math.prod(spatial_shape) != settings["patches_per_image"]:
        raise ValueError("inconsistent classifier spatial sampling configuration")
    indices = proportional_sample(train_labels, count, registered["sample_seed"])
    checkpoint = Path(cache_dir) / "classifier_moment.pt" if cache_dir is not None else None
    image_count, total = 0, None
    if checkpoint is not None and checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        image_count, total = state["images"], state["total"]
        if state["indices"] != indices or not 0 <= image_count <= count:
            raise ValueError("classifier calibration checkpoint differs from its training membership")
    subset = torch.utils.data.Subset(train_eval_dataset, indices[image_count:])
    moment, images, rows = _patch_moment(readout, batches(subset, batch_size), spatial_shape, total=total,
                                        image_count=image_count, checkpoint=checkpoint, indices=indices)
    if images != count or rows != settings["feature_rows"]:
        raise ValueError("classifier calibration image or patch count differs from the design")
    basis = carriers(moment.cpu(), [bits], [])["bases"][f"{bits}/Low"].contiguous()
    return {
        "weight_name": registered["weight_name"],
        "weight_shape": list(registered["weight_shape"]),
        "W0": readout.weight_matrix().cpu().double(),
        "E": basis,
        "q": q_vector(readout.outputs, registered["q_seed"], classification=True),
        "classification": True,
        "tolerance": 1e-10,
        "condition": condition,
        "sample_indices": indices,
        "feature_rows": rows,
    }


@torch.no_grad()
def insert_classifier_lessmark(model, assets, bits, alpha):
    """Apply one strength to a C_T copy and return the readout from its actual FP32 weights."""
    readout = _backbone_readout(model, assets["weight_name"], assets["weight_shape"])
    matrix = readout.weight_matrix()
    if not torch.equal(matrix.cpu().double(), assets["W0"].cpu()):
        raise ValueError("direct insertion requires a copy of the clean terminal carrier weights")
    payload = torch.as_tensor(bits)
    if payload.shape != (assets["E"].shape[1],) or not ((payload == 0) | (payload == 1)).all():
        raise ValueError("expected one binary information bit per carrier direction")
    lerra_insert(matrix, payload, {**assets, "alpha": alpha})
    readout.assign_matrix(matrix)
    return recover_classifier_lessmark(model, assets)


@torch.no_grad()
def recover_classifier_lessmark(model, assets):
    """Read the backbone using the original C_T weights, basis and output direction."""
    readout = _backbone_readout(model, assets["weight_name"], assets["weight_shape"])
    return decode(lerra_signal(readout.weight_matrix(), assets), erasure_at_zero=True)
