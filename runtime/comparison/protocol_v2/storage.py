from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Blocked(RuntimeError):
    def __init__(self, category, reason):
        self.category = category
        super().__init__(reason)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def state_hash(value):
    """Hash actual tensor/array state, without emitting model or key contents."""
    import numpy as np
    import torch
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            t=v.detach().cpu().contiguous()
            h.update(canonical(["tensor",str(t.dtype),list(t.shape)]).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(canonical(["ndarray",str(v.dtype),list(v.shape)]).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            h.update(b"mapping")
            for k in sorted(v,key=str): visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(canonical(["sequence",len(v)]).encode())
            for x in v: visit(x)
        elif isinstance(v,bytes):
            h.update(canonical(["bytes",len(v)]).encode());h.update(v)
        else:
            h.update(canonical(v.item() if isinstance(v,np.generic) else v).encode());h.update(b"\0")
    visit(value)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


@contextlib.contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        yield Path(temp)
        with open(temp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, value):
    with atomic_path(path) as p:
        p.write_text(canonical(value) + "\n")


def freeze_json(path,value):
    p=Path(path)
    if p.exists():
        if read_json(p)!=value:
            raise Blocked("resume_incompatible",f"immutable lock differs: {p}")
        return
    write_json(p,value)


def save_torch(path, value):
    import torch
    start=time.monotonic()
    with atomic_path(path) as p:
        # A file object avoids embedding the random temporary filename in the
        # torch ZIP prefix. Hash identity must not depend on temp-name randomness.
        with open(p,"wb") as f:
            torch.save(value, f)
    write_json(str(path)+".storage.json",{"phase":"serialization_and_atomic_commit","wall_seconds":time.monotonic()-start,"bytes":Path(path).stat().st_size,"path":str(path)})


def asset(path,value,asset_identity):
    """Immutable generated asset; re-use requires both origin identity and bytes."""
    import torch
    p=Path(path)
    receipt=p.with_suffix(p.suffix+".asset.json")
    if receipt.exists():
        r=read_json(receipt)
        if r["identity"]!=asset_identity:
            raise Blocked("resume_incompatible",f"asset origin changed: {p}")
        verify(r["artifact"])
        return torch.load(p,map_location="cpu",weights_only=False)
    if p.exists():
        raise Blocked("incomplete_asset",f"uncommitted asset file exists: {p}")
    save_torch(p,value)
    write_json(receipt,{"identity":asset_identity,"artifact":bind(p)})
    return value


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def verify(entry):
    p = Path(entry["path"])
    if not p.is_file():
        raise Blocked("missing_asset", str(p))
    if digest(p) != entry["sha256"]:
        raise Blocked("incompatible_asset", f"SHA256 mismatch: {p}")
    if p.suffix==".pt":
        with open(p,"rb") as f:
            is_pointer=f.read(64).lstrip().startswith(b"{")
        if is_pointer:
            pointer=read_json(p)
            if pointer.get("format")=="atomic_checkpoint_pointer_v1":
                child=Path(pointer["artifact"]["path"])
                if child.resolve()==p.resolve(): raise Blocked("incompatible_asset","checkpoint cannot point to itself")
                verify(pointer["artifact"])
    return p


def code_revision():
    # This workspace is not a git checkout. Content revision is explicit.
    files = sorted((ROOT / "protocol_v2").glob("*.py")) + [ROOT / "run_experiments.sh"]
    from importlib.metadata import version, PackageNotFoundError
    versions={}
    for name in ("torch","torchvision","numpy","scipy","pyldpc","numba","blobfile","pytorch-fid","matplotlib"):
        try: versions[name]=version(name)
        except PackageNotFoundError: versions[name]=None
    files += [ROOT/"configs/protocol_v2/sources.lock.json"]
    return identity({"files":{str(p.relative_to(ROOT)): digest(p) for p in files},"packages":versions})


def verify_sources(names=None):
    import subprocess
    for source in read_json(ROOT/"configs/protocol_v2/sources.lock.json"):
        if names is not None and source["name"] not in names:
            continue
        root=Path(source["path"])
        if not root.is_dir():
            raise Blocked("missing_asset",f"source unavailable: {source['name']}")
        actual=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        if actual!=source["revision"]:
            raise Blocked("incompatible_asset",f"source revision mismatch: {source['name']}")
        for relative,hash_ in source["files"].items():
            verify({"path":str(root/relative),"sha256":hash_})


def setup_root(path, required_bytes=1 << 30, create=True):
    root = Path(path).resolve()
    if root != Path("/data") and Path("/data") not in root.parents:
        raise Blocked("storage", "Large assets must be under /data; no home fallback")
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    writable = os.access(ancestor, os.W_OK)
    if create and not writable:
        raise Blocked("storage", f"Not writable: {ancestor}")
    free = shutil.disk_usage(ancestor).free
    if free < required_bytes:
        raise Blocked("resource", f"Need {required_bytes} bytes; available {free}")
    if create:
        for part in ("cache/torch", "cache/pip", "cache/huggingface", "cache/xdg", "cache/numba", "tmp", "assets", "data", "checkpoints", "runs/smoke", "runs/main"):
            (root / part).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=root):
            pass
    env = {"TORCH_HOME": "cache/torch", "PIP_CACHE_DIR": "cache/pip", "HF_HOME": "cache/huggingface", "XDG_CACHE_HOME": "cache/xdg", "NUMBA_CACHE_DIR": "cache/numba", "TMPDIR": "tmp"}
    for k, v in env.items():
        os.environ[k] = str(root / v)
    tempfile.tempdir = str(root / "tmp")
    return {"root": str(root), "writable": writable, "free_bytes": free, "required_bytes": required_bytes, "cache": {k: os.environ[k] for k in env}}


def declared_artifacts(value):
    """Collect immutable input declarations without logging their contents."""
    found={}
    if isinstance(value,dict):
        if isinstance(value.get("path"),str) and isinstance(value.get("sha256"),str):
            found[value["path"]]={k:value[k] for k in ("path","sha256")}
        for v in value.values():
            found.update({e["path"]:e for e in declared_artifacts(v)})
    elif isinstance(value,list):
        for v in value:
            found.update({e["path"]:e for e in declared_artifacts(v)})
    return list(found.values())


def complete(directory, job_identity, artifacts, result, inputs=()):
    for entry in inputs:
        verify(entry)
    write_json(Path(directory) / "completion.json", {"identity": job_identity, "artifacts": [bind(p) for p in artifacts], "inputs":list(inputs), "result": result, "complete": True})


def verified_completion(directory, job_identity):
    p = Path(directory) / "completion.json"
    if not p.exists():
        return None
    r = read_json(p)
    if r.get("identity") != job_identity or not r.get("complete") or not r.get("artifacts"):
        raise Blocked("resume_incompatible", f"Completion identity mismatch: {p}")
    for a in r["artifacts"]:
        verify(a)
        # verify also checks checkpoint pointer -> state blob, including external
        # shared-reference inputs; large torch ZIPs are not decoded as JSON.
    for a in r.get("inputs",[]):
        verify(a)
    return r


class Cost:
    def __init__(self, phase, device="cpu", output=None):
        self.phase, self.device = phase, str(device)
        self.output = output
        self.updates = self.images = 0

    def __enter__(self):
        import torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start = time.monotonic()
        return self

    def __exit__(self, *exc):
        import torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.monotonic() - self.start
        self.receipt = {"phase": self.phase, "wall_seconds": elapsed, "gpu_allocated_seconds": elapsed if self.device.startswith("cuda") else 0, "gpu_seconds_definition": "elapsed reservation, not measured SM active time", "peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.device.startswith("cuda") else 0, "optimizer_updates": self.updates, "processed_images": self.images, "device": self.device, "failed": exc[0] is not None}
        if self.output is not None:
            write_json(self.output, self.receipt)
