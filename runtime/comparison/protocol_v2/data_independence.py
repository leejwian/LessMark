"""Exact decoded-RGB separation before any classification training or selection."""
import hashlib
from pathlib import Path

from .storage import Blocked

RULE = "exact_rgb_test_priority_v1"


def raw_records(dataset, name, root):
    """Inspect original pixels, never stochastic transforms or model outputs."""
    import numpy as np
    from PIL import Image
    import torch
    if isinstance(dataset, torch.utils.data.Subset):
        rows = raw_records(dataset.dataset, name, root)
        return [rows[int(i)] for i in dataset.indices]
    result = []
    for i in range(len(dataset)):
        if hasattr(dataset, "records"):
            path, label = dataset.records[i]
            rid = Path(path).relative_to(root).as_posix()
            with Image.open(path) as source:
                pixels = np.asarray(source.convert("RGB"))
        else:
            pixels, label = dataset.data[i], int(dataset.targets[i])
            rid = f"{name}/official_{'train' if dataset.train else 'test'}/{i}"
        content = hashlib.sha256(("RGB:"+str(tuple(pixels.shape))+":").encode()+pixels.tobytes()).hexdigest()
        result.append({"id": rid, "label": int(label), "content_sha256": content})
    return result


def separate_records(splits):
    """Preserve test, remove its copies from validation, and all holdout copies
    from train. Conflicting labels in otherwise eligible train groups are all
    excluded. Holdout labels and model metrics never determine membership.
    """
    keys = ("train", "validation", "test")
    all_ids = [r["id"] for key in keys for r in splits[key]]
    if len(set(all_ids)) != len(all_ids):
        raise Blocked("incompatible_asset", "classification split record identities overlap")
    contents = {key: {r["content_sha256"] for r in splits[key]} for key in keys}
    groups = {}
    heldout = contents["validation"] | contents["test"]
    for row in splits["train"]:
        if row["content_sha256"] not in heldout:
            groups.setdefault(row["content_sha256"], set()).add(row["label"])
    conflicting = {digest for digest, labels in groups.items() if len(labels) > 1}
    kept, excluded = {}, {}
    for key in keys:
        kept[key], excluded[key] = [], []
        for i, row in enumerate(splits[key]):
            digest = row["content_sha256"]
            reasons = []
            if key != "test" and digest in contents["test"]:
                reasons.append("test_content")
            if key == "train" and digest in contents["validation"]:
                reasons.append("validation_content")
            if key == "train" and digest in conflicting:
                reasons.append("conflicting_training_labels")
            if reasons:
                excluded[key].append({"position": i, "id": row["id"], "content_sha256": digest, "reasons": reasons})
            else:
                kept[key].append(i)
        if not kept[key]:
            raise Blocked("incompatible_asset", "exact RGB exclusion emptied classification "+key)
    if {splits["train"][i]["label"] for i in kept["train"]} != {r["label"] for r in splits["train"]}:
        raise Blocked("incompatible_asset", "exact RGB exclusion removed an entire training class")
    remaining = {key: {splits[key][i]["content_sha256"] for i in kept[key]} for key in keys}
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    overlaps = {a+"/"+b: len(remaining[a] & remaining[b]) for a, b in pairs}
    if any(overlaps.values()):
        raise Blocked("incompatible_asset", "exact RGB separation failed")
    return {"rule": RULE, "kept_positions": kept, "excluded": excluded,
            "counts": {key: {"original": len(splits[key]), "retained": len(kept[key]),
                             "excluded": len(excluded[key])} for key in keys},
            "remaining_content_overlap_groups": overlaps,
            "record_disjoint": True, "exact_rgb_content_disjoint": True,
            "near_duplicate_independence_claimed": False,
            "heldout_use": "content exclusion only; no heldout labels or model metrics"}
