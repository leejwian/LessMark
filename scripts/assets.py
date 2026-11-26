#!/usr/bin/env python3
"""List or verify the external, immutable inputs of the registered experiment."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "experiment_2/config"
PROTOCOL = Path("/data/lerra_parameter_watermark_protocol")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Inventory:
    def __init__(self, root):
        self.root = root
        self.files = {}
        self.git_sources = {}

    def add(self, path, digest=None, size=None):
        path = Path(path)
        relative = path.relative_to(PROTOCOL)
        if ".." in relative.parts or relative.parts[0] not in ("assets", "checkpoints", "data", "runs", "sources"):
            raise ValueError(f"unexpected input path: {path}")
        name = path.relative_to("/data")
        previous_digest, previous_size = self.files.get(name, (None, None))
        if ((previous_digest and digest and previous_digest != digest)
                or (previous_size is not None and size is not None and previous_size != size)):
            raise ValueError(f"conflicting records for {path}")
        self.files[name] = (digest or previous_digest, size if size is not None else previous_size)
        return self.root / name

    def bind(self, record):
        return self.add(record["path"], record["sha256"], record.get("bytes"))

    def load(self, record):
        path = self.bind(record)
        if sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        return json.loads(path.read_bytes())

    def model(self, model):
        source = model["source"]
        for binding in source["files"].values():
            self.bind(binding)
        if "source_revision.json" not in source["files"]:
            self.git_sources[Path(source["path"])] = source["revision"]
        for binding in model.get("vae", {}).get("files", {}).values():
            self.bind(binding)
        for binding in model.get("input_manifests", {}).values():
            self.bind(binding)
        adaptation = model.get("dataset_adaptation", {})
        for key in ("T_evidence", "training_membership", "validation_membership"):
            if key in adaptation:
                self.bind(adaptation[key])

    def git_objects(self):
        # Only repository objects and local revision pointers are needed by
        # verify_source. Do not copy a remote URL, hooks or account settings.
        for canonical, revision in self.git_sources.items():
            repo = self.root / canonical.relative_to("/data")
            actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD^{commit}"], text=True).strip()
            if actual != revision:
                raise ValueError(f"upstream revision mismatch: {repo}")
            gitdir = repo / ".git"
            for pattern in ("HEAD", "shallow", "packed-refs", "refs/heads/**/*", "objects/**/*"):
                for item in gitdir.glob(pattern):
                    if item.is_file() and "info" not in item.relative_to(gitdir).parts:
                        self.add(canonical / ".git" / item.relative_to(gitdir))

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


def collect(root):
    inventory = Inventory(root)
    design = json.loads((CONFIG / "NEXT_EXPERIMENT_DESIGN.json").read_bytes())
    embedding = design["embedding"]
    for condition in embedding["fixed_source_checkpoints"].values():
        inventory.bind(condition["checkpoint"])

    cifar = design["data"]["source_CIFAR100"]["source_files"]
    raw = Path(cifar["raw_data_root"])
    for record in cifar["official_python_files"]:
        inventory.add(raw / record["path_from_raw_data_root"], record["sha256"])
    inventory.add(raw / "cifar-100-python/meta", "a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9")

    photos = embedding["lessmark_real_image_calibration"]["photo_lists"]
    pools = embedding["native_generator_training"]["train_pools"]
    for domain in ("imagenet", "caltech256"):
        for listing in (photos[domain], pools[domain]["manifest"]):
            for row in inventory.load(listing):
                inventory.bind(row)
    registration = Path(pools["caltech256"]["manifest"]["path"]).with_name("result.json")
    splits = json.loads(inventory.add(registration).read_bytes())["manifests"]
    for key in ("reference_validation", "main_heldout"):
        inventory.bind(splits[key])

    binding = design["strength"]["historical_centers_source"]
    history = json.loads((CONFIG / binding["path_relative_to_this_file"]).read_bytes())
    for row in history["rows"]:
        inventory.bind(row["selection"])
        if row["method"] != "lerra":
            continue
        contract = inventory.load(row["contract"])
        if "model" not in contract:
            continue
        model = contract["model"]
        inventory.model(model)
        if model["kind"] == "styleswin_256":
            # Historical manifests supply per-file identities for all 70k FFHQ
            # PNGs. Experiment 2 uses the registered 60k official training IDs.
            low, high = pools["ffhq"]["official_training_ids_inclusive"]
            base = Path(pools["ffhq"]["image_directory"])
            wanted = {base / f"{i:05d}.png" for i in range(low, high + 1)}
            for manifest in model["real_evaluation_memberships"].values():
                for entry in inventory.load(manifest):
                    path = Path(entry["path"])
                    if path in wanted:
                        inventory.bind(entry)
                        wanted.remove(path)
            if wanted:
                raise ValueError(f"FFHQ inventory lacks {len(wanted)} registered training IDs")

    scope = design["robustness"]["eligible_scope"]["generation_reference_rows"]
    historical_results = json.loads((PROJECT / scope["path_from_project_root"]).read_bytes())
    for row in historical_results["rows"]:
        if row["method"] == "lerra" and row["modification"] == "prune30":
            inventory.bind(row["source_result"])
    inventory.git_objects()
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=os.environ.get("LERRA_DATA_ROOT", "/data"),
                        help="local directory replacing /data, containing lerra_parameter_watermark_protocol/")
    parser.add_argument("--files-from", type=Path, metavar="FILE", help="write rsync --files-from input")
    parser.add_argument("--verify", action="store_true", help="also hash every file with a registered SHA-256")
    args = parser.parse_args()
    try:
        inventory = collect(args.root.expanduser().resolve())
        total = inventory.check(args.verify)
        if args.files_from:
            args.files_from.write_text("".join(f"{name.as_posix()}\n" for name in sorted(inventory.files)),
                                       encoding="utf-8")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Asset inventory failed: {error}\n")
    print(f"{len(inventory.files):,} files, {total:,} bytes ({total / 2**30:.1f} GiB)", file=sys.stderr)


if __name__ == "__main__":
    main()
