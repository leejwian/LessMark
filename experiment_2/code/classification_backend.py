"""CIFAR-100 source-task training and paired validation for experiment 2."""

import copy
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100

from runtime.comparison.protocol_v2.data import batches, build_model, transforms
from runtime.comparison.protocol_v2.storage import read_json, state_hash, verify, write_json
from runtime.comparison.protocol_v2.training import (
    UpdateBatches, classification_loss, lr_at, optimizer_for, seed_all,
)


PROJECT = Path(__file__).resolve().parents[2]


def _bound_file(binding, base, field):
    return verify({"path": str(base / binding[field]), "sha256": binding["sha256"]})


class ClassificationBackend:
    def __init__(self, design, condition, device, recipe):
        self.condition = condition
        self.design = design
        self.device = torch.device(device)
        self.recipe = copy.deepcopy(recipe)
        source = design["data"]["source_CIFAR100"]
        inputs = source["source_files"]
        bound_split = inputs["validation_and_official_split"]
        self.split_sha256 = bound_split["sha256"]
        split = read_json(_bound_file(bound_split, PROJECT, "path_from_project_root"))
        membership = read_json(_bound_file(inputs["retained_membership_and_rgb_hashes"], PROJECT,
                                           "path_from_project_root"))
        raw_root = Path(inputs["raw_data_root"])
        _bound_file(inputs["official_python_files"][0], raw_root, "path_from_raw_data_root")
        # Use a single copy of the raw images. Different views apply random
        # training augmentation or deterministic validation preprocessing.
        source_train = CIFAR100(raw_root, train=True, download=False, transform=transforms(split["preprocessing"], True))
        source_eval = copy.copy(source_train)
        source_eval.transform = transforms(split["preprocessing"], False)
        if split["record_ids"] != [f"cifar100/official_train/{i}" for i in range(len(source_train))]:
            raise ValueError("CIFAR input order disagrees with the registered membership")
        for split_name in ("train", "validation"):
            ids = split["indices"][split_name]
            retained = [membership["records"][split_name][i]["id"] for i in membership["kept_positions"][split_name]]
            if len(ids) != source["counts"][split_name] or retained != [split["record_ids"][i] for i in ids]:
                raise ValueError(f"CIFAR {split_name} membership disagrees with the registered split")
        train_ids = split["indices"]["train"]
        self.train_eval_dataset = Subset(source_eval, train_ids)
        self.train_labels = [source_train.targets[i] for i in train_ids]
        self.validation = Subset(source_eval, split["indices"]["validation"])
        self.batcher = UpdateBatches(Subset(source_train, train_ids), self.recipe["microbatch"],
                                     self.recipe["accumulation"], self.recipe["seed"])
        schedule = design["checkpoint_schedule"]
        if self.batcher.updates_per_epoch != schedule["classification_updates_per_epoch"]:
            raise ValueError("classifier effective batch size differs from the registered updates per epoch")
        self.schedule = dict(self.recipe["schedule"], horizon_updates=schedule["classification_terminal_updates"])
        milestones = self.schedule.pop("milestones_epochs")
        self.schedule["milestones_updates"] = [epoch * self.batcher.updates_per_epoch for epoch in milestones]

        archive = design["robustness"]["eligible_scope"]["classifier_layer_lists"][condition]
        original = read_json(_bound_file(archive, PROJECT, "path_from_project_root"))["config"]
        if original["data"]["preprocessing"] != split["preprocessing"]:
            raise ValueError("registered classifier and source split disagree on preprocessing")
        self.model = build_model(original["model"]).to(device=self.device, dtype=torch.float32)
        checkpoint = design["embedding"]["fixed_source_checkpoints"][condition]["checkpoint"]
        self.initial_weights = torch.load(verify(checkpoint), map_location="cpu", weights_only=True)
        self.reset_training()

    def reset_training(self):
        """Restore the same C0 and reset every parameter, buffer and optimizer for a new arm."""
        self.model.load_state_dict(self.initial_weights, strict=True)
        self.model.requires_grad_(True).train()
        self.model.zero_grad(set_to_none=True)
        self.optimizer = optimizer_for(self.model, self.recipe["optimizer"])
        seed_all(self.recipe["seed"])

    def train_step(self, step, watermark_loss=None):
        """One optimizer update. The optional callback returns the weighted parameter-only loss."""
        self.model.train()
        lr = lr_at(step, self.schedule)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = self.batcher.accumulation
        total_ce = torch.zeros((), device=self.device)
        for micro in range(accumulation):
            # The absolute step and microbatch select the same images, augmentation
            # and dropout RNG on every arm and after restoring a checkpoint.
            batch = self.batcher.get(step, micro)
            ce = classification_loss(self.model, batch, self.device)["total"]
            (ce / accumulation).backward()
            total_ce.add_(ce.detach() / accumulation)
        wm = total_ce.new_zeros(())
        if watermark_loss is not None:
            # The four losses depend only on weights and fixed assets, so the
            # averaged gradient requires exactly one evaluation per update.
            wm = watermark_loss()
            wm.backward()
        if not torch.isfinite(total_ce + wm.detach()):
            raise FloatingPointError("nonfinite classifier objective")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe["gradient_clip"], error_if_nonfinite=True)
        self.optimizer.step()
        return {"cross_entropy": total_ce.item(), "weighted_watermark_loss": wm.detach().item(), "lr": lr}

    def training_state_dict(self):
        return {"optimizer": self.optimizer.state_dict()}

    def load_training_state_dict(self, state):
        # Data order, dropout, augmentations and LR are rebuilt from the next
        # absolute update; momentum is the only persistent auxiliary state.
        self.optimizer.load_state_dict(state["optimizer"])

    @torch.inference_mode()
    def _validation_counts(self):
        self.model.eval()
        correct = torch.zeros((), dtype=torch.int64, device=self.device)
        invalid = torch.zeros_like(correct)
        for inputs, labels in batches(self.validation, self.recipe["validation_batch"]):
            logits = self.model(inputs.to(self.device))
            if logits.shape != (len(labels), 100) or not logits.is_floating_point():
                raise ValueError("expected floating-point CIFAR-100 logits for each validation image")
            valid = logits.isfinite().all(dim=1)
            invalid += (~valid).sum()
            correct += (valid & (logits.argmax(1) == labels.to(self.device))).sum()
        invalid_count = int(invalid.item())
        return {"invalid_examples": invalid_count, "correct": None if invalid_count else int(correct.item())}

    def prepare_quality(self, cache_dir):
        """Compute clean validation accuracy once, including recovery after a partial cache write."""
        path = Path(cache_dir) / "classifier_clean_quality.json"
        identity = {"condition": self.condition, "validation_split_sha256": self.split_sha256,
                    "examples": len(self.validation), "clean_weights_sha256": state_hash(self.model.state_dict())}
        if path.is_file():
            cached = read_json(path)
            if any(cached.get(k) != v for k, v in identity.items()) or not self._valid_count(cached):
                raise ValueError("incompatible classifier validation cache")
            return
        write_json(path, {**identity, **self._validation_counts()})

    def _valid_count(self, cached):
        # Earlier caches were written only after verifying that all logits were finite.
        invalid = cached.get("invalid_examples", 0)
        if type(invalid) is not int or not 0 <= invalid <= len(self.validation):
            return False
        if invalid:
            return "correct" in cached and cached["correct"] is None
        return type(cached.get("correct")) is int and 0 <= cached["correct"] <= len(self.validation)

    def quality(self, cache_dir):
        cached = read_json(Path(cache_dir) / "classifier_clean_quality.json")
        count = len(self.validation)
        if (cached.get("condition") != self.condition or cached.get("validation_split_sha256") != self.split_sha256
                or cached.get("examples") != count or not self._valid_count(cached)):
            raise ValueError("classifier validation cache disagrees with this condition")
        measured = self._validation_counts()
        clean, marked = cached["correct"], measured["correct"]
        complete = clean is not None and marked is not None
        return {"metric": "source_validation_top1", "examples": count, "clean_correct": clean,
                "marked_correct": marked, "quality_status": "ok" if complete else "undefined",
                "clean_invalid_examples": cached.get("invalid_examples", 0),
                "marked_invalid_examples": measured["invalid_examples"],
                "clean_top1_pct": 100 * clean / count if clean is not None else None,
                "marked_top1_pct": 100 * marked / count if marked is not None else None,
                "accuracy_drop_pp": 100 * (clean - marked) / count if complete else None}
