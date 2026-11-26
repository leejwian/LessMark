from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .storage import Blocked, identity, freeze_json, bind
from .data_independence import raw_records, separate_records

HEADS = {"resnet18": "fc", "resnet34": "fc", "wide_resnet50_2": "fc", "alexnet": "classifier.6", "vgg13": "classifier.6", "vgg16": "classifier.6", "vit_b_16": "heads.head", "swin_v2_b": "head", "swin_v2_s": "head", "googlenet": "fc", "mobilenet_v3_large": "classifier.3"}
CLASSES = {"cifar10": 10, "cifar100": 100, "caltech101": 101, "caltech256": 256, "tinyimagenet": 200}


def build_model(spec):
    from torchvision import models
    name = spec["architecture"]
    if name not in HEADS or spec["implementation"] != "torchvision":
        raise Blocked("unsupported_configuration", "supported, version-pinned model implementation: torchvision")
    kwargs = {"weights": None, "num_classes": spec["classes"]}
    if "width_mult" in spec:
        width = spec["width_mult"]
        if name != "mobilenet_v3_large":
            raise Blocked("unsupported_configuration", "width_mult is only supported for MobileNetV3-Large")
        if isinstance(width, bool) or not isinstance(width, (int, float)) or not math.isfinite(width) or width <= 0:
            raise Blocked("unsupported_configuration", "MobileNetV3 width_mult must be a finite positive number")
        kwargs["width_mult"] = float(width)
    if name == "googlenet":
        kwargs.update(aux_logits=False, init_weights=False)
    model = getattr(models, name)(**kwargs)
    if spec.get("cifar_stem", False):
        if name != "resnet18":
            raise ValueError("cifar_stem adapter is only defined for ResNet18")
        model.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
    if name in ("alexnet", "vgg13", "vgg16"):
        from .deterministic_pooling import install_identity_pool_backward
        install_identity_pool_backward(model, name)
    model._protocol_model_architecture = name
    return model


def transforms(spec, train):
    from torchvision import transforms as T
    if spec is None:
        raise Blocked("research_setting", "preprocessing must match checkpoint and task recipe")
    if train:
        ops = [T.RandomResizedCrop(spec["crop"], scale=tuple(spec["scale"]), ratio=tuple(spec["ratio"]), antialias=True), T.RandomHorizontalFlip(spec["flip"])]
    else:
        ops = [T.Resize(spec["resize"], antialias=True), T.CenterCrop(spec["crop"])]
    return T.Compose([T.Lambda(lambda im: im.convert("RGB")), *ops, T.ToTensor(), T.Normalize(spec["mean"], spec["std"])])


class ImageRecords(torch.utils.data.Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = records, transform
        self.targets = [r[1] for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        p,y = self.records[i]
        with Image.open(p) as im:
            return self.transform(im.convert("RGB")), y


def folder_records(root, dataset, test=False):
    root = Path(root)
    if dataset == "tinyimagenet":
        classes = sorted((root/"wnids.txt").read_text().split())
        mapping = {c:i for i,c in enumerate(classes)}
        if test:
            return [(str(root/"val/images"/row[0]), mapping[row[1]]) for line in (root/"val/val_annotations.txt").read_text().splitlines() if (row := line.split())]
        return [(str(p), mapping[c]) for c in classes for p in sorted((root/"train"/c/"images").glob("*.JPEG"))]
    exclude = {"BACKGROUND_Google", "257.clutter"}
    classes = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name not in exclude]
    if len(classes) != CLASSES[dataset]:
        raise ValueError("class mapping mismatch (background/clutter excluded)")
    return [(str(p),i) for i,c in enumerate(classes) for p in sorted(c.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]


def split_indices(labels, dataset):
    y = np.asarray(labels)
    rng = np.random.default_rng(20261003)
    result = {"train": [], "validation": [], "test": []}
    caltech = dataset.startswith("caltech")
    for c in sorted(np.unique(y)):
        indices = rng.permutation(np.flatnonzero(y == c))
        ntrain = math.floor(len(indices)*(.7 if caltech else .9))
        nval = math.floor(len(indices)*.15) if caltech else len(indices)-ntrain
        result["train"].extend(indices[:ntrain].tolist())
        result["validation"].extend(indices[ntrain:ntrain+nval].tolist())
        result["test"].extend(indices[ntrain+nval:].tolist())
    return result


def evaluation_membership(records, policy, dataset):
    """Select the same 30 official held-out examples per CIFAR100 class."""
    expected={'count':3000,'seed':20261023,'strategy':'stratified_equal_per_class'}
    if dataset!='cifar100' or policy!=expected:
        raise ValueError('classification evaluation sampling requires the frozen CIFAR100 3000-image policy')
    labels=np.asarray([r['label'] for r in records])
    if set(labels.tolist())!=set(range(100)):
        raise ValueError('evaluation population must contain all 100 CIFAR100 classes')
    rng=np.random.default_rng(policy['seed']);indices=[]
    for label in range(100):
        candidates=np.flatnonzero(labels==label)
        if len(candidates)<30:
            raise ValueError('fewer than 30 retained test examples in a CIFAR100 class')
        indices.extend(rng.choice(candidates,30,replace=False).tolist())
    indices.sort()  # Preserve official-test order after selecting membership.
    return {'schema':'classification_evaluation_membership_v1','dataset':dataset,'policy':dict(policy),
            'source_split':'test_after_exact_rgb_independence','available_examples':len(records),
            'source_records_identity':identity(records),'selected_positions':indices,
            'selected_records':[records[i] for i in indices],
            'per_class':30,'training_and_validation_unchanged':True}


def classification_data(spec, output):
    from torchvision.datasets import CIFAR10, CIFAR100
    name, root = spec["name"], Path(spec["path"])
    separate_evaluation=spec.get("selection_split")=="validation" and spec.get("evaluation_split")=="test"
    if separate_evaluation and spec.get("evaluation_sampling") is not None:
        raise ValueError("separate final evaluation uses the complete test population without sampling")
    if not root.exists():
        raise Blocked("missing_asset", f"dataset unavailable: {root}")
    tr, ev = transforms(spec["preprocessing"], True), transforms(spec["preprocessing"], False)
    if name in ("cifar10", "cifar100"):
        cls = CIFAR10 if name == "cifar10" else CIFAR100
        base = cls(root, train=True, download=False, transform=tr)
        evalbase = cls(root, train=True, download=False, transform=ev)
        test = cls(root, train=False, download=False, transform=ev)
        labels = base.targets
        record_ids = [f"{name}/official_train/{i}" for i in range(len(base))]
    else:
        rec = folder_records(root, name)
        base, evalbase = ImageRecords(rec,tr), ImageRecords(rec,ev)
        labels = base.targets
        record_ids = [p for p,_ in rec]
        test = ImageRecords(folder_records(root,name,True),ev) if name == "tinyimagenet" else None
    ids = split_indices(labels,name)
    # One raw-pixel pass over the base avoids repeatedly decoding it for each
    # Subset. The fixed RNG partition is retained except for exact exclusions.
    original_ids = {key: list(value) for key, value in ids.items()}
    records = raw_records(evalbase, name, root)
    splits = {key: [records[i] for i in ids[key]] for key in ids}
    if test is not None:
        splits["test"] = raw_records(test, name, root)
    independence = separate_records(splits)
    for key in ("train", "validation"):
        ids[key] = [ids[key][i] for i in independence["kept_positions"][key]]
    if test is None:
        test = torch.utils.data.Subset(evalbase, ids["test"])
    evaluation=None
    if spec.get('evaluation_sampling') is not None:
        # Keep the full held-out population in exact-content exclusion above:
        # reducing evaluation work must never return test duplicates to train.
        retained=independence['kept_positions']['test']
        evaluation=evaluation_membership([splits['test'][i] for i in retained],spec['evaluation_sampling'],name)
        test=torch.utils.data.Subset(test,[retained[i] for i in evaluation['selected_positions']])
        freeze_json(Path(output)/'evaluation_membership.json',evaluation)
    # Hash actual input bytes, not merely names/labels. Cost is preparation and
    # may be substantial for large datasets; this is never invoked by smoke.
    if name in ("cifar10", "cifar100"):
        files = sorted((root/base.base_folder).rglob("*"))
    else:
        files = [Path(p) for p in record_ids]
        if name == "tinyimagenet":
            files += [Path(p) for p,_ in folder_records(root,name,True)]
            files += [root/"wnids.txt",root/"val/val_annotations.txt"]
    inputs = [bind(p) for p in files if p.is_file()]
    # Bind the complete membership/content receipt without including the output
    # path in dataset identity; input checks and workers must identify the same data.
    evidence = {**independence, "dataset": name, "input_files": inputs,
                "records": {key: [{k: r[k] for k in ("id", "content_sha256")} for r in rows]
                            for key, rows in splits.items()}}
    freeze_json(Path(output)/"data_independence_manifest.json", evidence)
    manifest = {"split_seed": 20261003, "dataset": name, "input_files": inputs, "record_ids": record_ids, "indices": ids, "original_indices": original_ids,
                "independence": {"rule": independence["rule"], "evidence_identity": identity(evidence), "counts": independence["counts"]},
                "preprocessing": spec["preprocessing"], "public_checkpoint_validation_independence": spec.get("validation_independence")}
    if separate_evaluation:
        manifest["selection_split"]="validation"
        manifest["evaluation_split"]="test"
        manifest["test_population"]="full official test split" if name in ("cifar10","cifar100") else "full declared test split"
        manifest["historical_test_exposure"]=True
    if evaluation is not None:
        manifest['evaluation_sampling']={'policy':dict(spec['evaluation_sampling']),
            'membership_identity':identity(evaluation),'evaluated_examples':len(test)}
    freeze_json(Path(output)/"data_split_manifest.json", manifest)
    return {"train": torch.utils.data.Subset(base,ids["train"]), "train_eval": torch.utils.data.Subset(evalbase,ids["train"]), "validation": torch.utils.data.Subset(evalbase,ids["validation"]), "test": test, "train_labels": [labels[i] for i in ids["train"]], "identity": identity(manifest),
            "independence": bind(Path(output)/"data_independence_manifest.json"),
            **({'evaluation_membership':bind(Path(output)/'evaluation_membership.json')} if evaluation is not None else {})}


def batches(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
