"""Read immutable registered manifests using the current server's data mount."""

import json
import os
from pathlib import Path


def data_root():
    mount = Path(os.environ.get("LERRA_DATA_ROOT", "/data")).expanduser()
    if not mount.is_absolute():
        raise ValueError("LERRA_DATA_ROOT must be an absolute path to the copy of /data")
    return mount


def input_path(canonical_path):
    """Resolve one original input path. Call once, at the registered-input boundary."""
    path = Path(canonical_path)
    if path.is_relative_to("/data"):
        return data_root() / path.relative_to("/data")
    return path


def read_registered_json(local_path):
    """Open a local manifest, then translate its original absolute paths in memory.

    Callers verify the original file bytes before calling this for a hashed input.
    Relative paths, hashes and all other scientific settings retain their values.
    """
    with Path(local_path).open(encoding="utf-8") as stream:
        record = json.load(stream)
    mount = data_root()
    if mount == Path("/data"):
        return record

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and (value.startswith("/data/") or value == "/data"):
            return str(mount / Path(value).relative_to("/data"))
        return value

    return relocate(record)
