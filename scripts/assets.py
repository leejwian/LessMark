#!/usr/bin/env python3
"""Check the local strength-selection inputs or write their file list for rsync."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "experiment_2/config"
REGISTERED_ROOT = Path("/data/lerra_parameter_watermark_protocol")
INPUT_DIRECTORIES = {"assets", "checkpoints", "data", "runs", "sources"}
CIFAR_META_SHA256 = "a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Inventory:
    """Track files relative to the data mount, merging duplicate manifest entries."""

    def __init__(self, root):
        self.root = root
        self.files = {}
        self.git_sources = {}

    def add(self, path, digest=None, size=None):
        path = Path(path)
        relative = path.relative_to(REGISTERED_ROOT)
        if not relative.parts or ".." in relative.parts or relative.parts[0] not in INPUT_DIRECTORIES:
            raise ValueError(f"unexpected input path: {path}")

        name = path.relative_to(REGISTERED_ROOT.parent)
        previous_digest, previous_size = self.files.get(name, (None, None))
        if digest and previous_digest and digest != previous_digest:
            raise ValueError(f"conflicting records for {path}")
        if size is not None and previous_size is not None and size != previous_size:
            raise ValueError(f"conflicting records for {path}")
        self.files[name] = (digest or previous_digest, size if size is not None else previous_size)
        return self.root / name

    def add_record(self, record):
        return self.add(record["path"], record["sha256"], record.get("bytes"))

    def read_manifest(self, record):
        path = self.add_record(record)
        if sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        return json.loads(path.read_bytes())

    def add_generator(self, model):
        source = model["source"]
        for record in source["files"].values():
            self.add_record(record)
        if "source_revision.json" not in source["files"]:
            self.git_sources[Path(source["path"])] = source["revision"]
        for record in model.get("vae", {}).get("files", {}).values():
            self.add_record(record)
        for record in model.get("input_manifests", {}).values():
            self.add_record(record)
        adaptation = model.get("dataset_adaptation", {})
        for key in ("T_evidence", "training_membership", "validation_membership"):
            if key in adaptation:
                self.add_record(adaptation[key])

    def add_git_revisions(self):
        # verify_source needs the original commit and its objects on the target.
        patterns = ("HEAD", "shallow", "packed-refs", "refs/heads/**/*", "objects/**/*")
        for source, revision in self.git_sources.items():
            repo = self.root / source.relative_to(REGISTERED_ROOT.parent)
            command = ["git", "-C", str(repo), "rev-parse", "HEAD^{commit}"]
            actual = subprocess.check_output(command, text=True).strip()
            if actual != revision:
                raise ValueError(f"upstream revision mismatch: {repo}")
            gitdir = repo / ".git"
            for pattern in patterns:
                for item in gitdir.glob(pattern):
                    relative = item.relative_to(gitdir)
                    if item.is_file() and "info" not in relative.parts:
                        self.add(source / ".git" / relative)

    def check(self, verify):
        total = 0
        for name in sorted(self.files):
            digest, expected_size = self.files[name]
            path = self.root / name
            size = path.stat().st_size
            if not path.is_file() or (expected_size is not None and size != expected_size):
                raise ValueError(f"wrong file or length: {path}")
            if verify and digest and sha256(path) != digest:
                raise ValueError(f"SHA-256 mismatch: {path}")
            total += size
        return total


def add_cifar100(inventory, source_files):
    raw = Path(source_files["raw_data_root"])
    for record in source_files["official_python_files"]:
        inventory.add(raw / record["path_from_raw_data_root"], record["sha256"])
    inventory.add(raw / "cifar-100-python/meta", CIFAR_META_SHA256)


def add_imagenet_and_caltech256(inventory, calibration_lists, train_pools):
    for dataset in ("imagenet", "caltech256"):
        for manifest in (calibration_lists[dataset], train_pools[dataset]["manifest"]):
            for photo in inventory.read_manifest(manifest):
                inventory.add_record(photo)

    train_manifest = Path(train_pools["caltech256"]["manifest"]["path"])
    registration = inventory.add(train_manifest.with_name("result.json"))
    splits = json.loads(registration.read_bytes())["manifests"]
    for split in ("reference_validation", "main_heldout"):
        inventory.add_record(splits[split])


def add_ffhq_training_images(inventory, pool, memberships):
    """Take the 60,000 registered training IDs from historical 70,000-image lists."""
    first, last = pool["official_training_ids_inclusive"]
    directory = Path(pool["image_directory"])
    missing = {directory / f"{image_id:05d}.png" for image_id in range(first, last + 1)}
    for manifest in memberships.values():
        for photo in inventory.read_manifest(manifest):
            path = Path(photo["path"])
            if path in missing:
                inventory.add_record(photo)
                missing.remove(path)
    if missing:
        raise ValueError(f"FFHQ inventory lacks {len(missing)} registered training IDs")


def add_selection_records_and_models(inventory, history_file, ffhq_pool):
    history = json.loads(history_file.read_bytes())
    for row in history["rows"]:
        inventory.add_record(row["selection"])
        # The LessMark row supplies the shared source contract for each condition.
        if row["method"] != "lerra":
            continue
        contract = inventory.read_manifest(row["contract"])
        if "model" not in contract:
            continue
        model = contract["model"]
        inventory.add_generator(model)
        if model["kind"] == "styleswin_256":
            add_ffhq_training_images(inventory, ffhq_pool, model["real_evaluation_memberships"])


def collect(root):
    inventory = Inventory(root)
    design = json.loads((CONFIG / "NEXT_EXPERIMENT_DESIGN.json").read_bytes())
    embedding = design["embedding"]
    for condition in embedding["fixed_source_checkpoints"].values():
        inventory.add_record(condition["checkpoint"])

    pools = embedding["native_generator_training"]["train_pools"]
    calibration_lists = embedding["lessmark_real_image_calibration"]["photo_lists"]
    add_cifar100(inventory, design["data"]["source_CIFAR100"]["source_files"])
    add_imagenet_and_caltech256(inventory, calibration_lists, pools)

    history = design["strength"]["historical_centers_source"]
    add_selection_records_and_models(inventory, CONFIG / history["path_relative_to_this_file"], pools["ffhq"])

    scope = design["robustness"]["eligible_scope"]["generation_reference_rows"]
    historical_results = json.loads((PROJECT / scope["path_from_project_root"]).read_bytes())
    for row in historical_results["rows"]:
        if row["method"] == "lerra" and row["modification"] == "prune30":
            inventory.add_record(row["source_result"])

    inventory.add_git_revisions()
    return inventory


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="To transfer these inputs, run ./lessmark copy-inputs on the source server. See docs/assets.md.",
    )
    parser.add_argument(
        "--root", type=Path, default=os.environ.get("LERRA_DATA_ROOT", "/data"),
        help="local data mount containing lerra_parameter_watermark_protocol/ (default: LERRA_DATA_ROOT or /data)",
    )
    parser.add_argument(
        "--files-from", type=Path, metavar="FILE",
        help="write checked paths, relative to --root, for rsync --files-from=FILE",
    )
    parser.add_argument("--verify", action="store_true", help="also hash every file with a registered SHA-256")
    args = parser.parse_args()
    try:
        inventory = collect(args.root.expanduser().resolve())
        total = inventory.check(args.verify)
        if args.files_from:
            with args.files_from.open("w", encoding="utf-8") as output:
                for name in sorted(inventory.files):
                    print(name.as_posix(), file=output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Asset inventory failed: {error}\n")
    print(f"{len(inventory.files):,} files, {total:,} bytes ({total / 2**30:.1f} GiB)", file=sys.stderr)


if __name__ == "__main__":
    main()
