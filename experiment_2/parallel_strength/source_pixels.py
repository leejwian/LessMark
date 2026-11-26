"""Share deterministic training crops across U-Net and SiT arms; keep per-draw flips native."""
import fcntl
import hashlib
import io
import os
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import torch

from runtime.comparison.protocol_v2 import rgb
from runtime.comparison.protocol_v2.generative_reference import _seed
from runtime.comparison.protocol_v2.storage import atomic_path, freeze_json

# Each object binds its source SHA-256 and its unnormalized CHW pixel SHA-256.
SHAPE = (3, 256, 256)
PIXEL_BYTES = 3 * 256 * 256
HEADER_BYTES = 64


def _stamp(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class _DamagedCrop(ValueError):
    pass


class SourcePixels:
    def __init__(self, photos, directory):
        self.rows = photos.rows
        self.directory = Path(directory)
        self.sources_checked = {}
        self.crops_checked = {}
        origin = {
            'schema': 'native_center_crop_chw_uint8_v1',
            'transform': 'runtime.comparison.protocol_v2.rgb.center_crop(size=256), before source flip',
            'transform_module_sha256': hashlib.sha256(Path(rgb.__file__).read_bytes()).hexdigest(),
            'numpy': np.__version__, 'pillow': PIL.__version__, 'shape': list(SHAPE),
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / '.origin.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            freeze_json(self.directory / 'origin.json', origin)

    def _source_bytes(self, index, row):
        source = Path(row['path'])
        with source.open('rb') as stream:
            before = _stamp(os.fstat(stream.fileno()))
            contents = stream.read()
            after = _stamp(os.fstat(stream.fileno()))
        if (before != after or after != _stamp(source.stat())
                or ('bytes' in row and len(contents) != row['bytes'])
                or hashlib.sha256(contents).hexdigest() != row['sha256']):
            raise ValueError(f'Registered training photo changed: {source}')
        self.sources_checked[index] = after
        return contents

    def _read(self, path, source_digest):
        with path.open('rb') as stream:
            stat = _stamp(os.fstat(stream.fileno()))
            if stat[2] != HEADER_BYTES + PIXEL_BYTES:
                raise _DamagedCrop(f'Wrong cached crop size at {path}')
            contents = stream.read()
            if stat != _stamp(os.fstat(stream.fileno())) or len(contents) != HEADER_BYTES + PIXEL_BYTES:
                raise _DamagedCrop(f'Cached crop changed during read: {path}')
        if contents[:32] != source_digest:
            raise _DamagedCrop(f'Wrong source for cached crop at {path}')
        pixels = memoryview(contents)[HEADER_BYTES:]
        if self.crops_checked.get(source_digest) != stat:
            if hashlib.sha256(pixels).digest() != contents[32:HEADER_BYTES]:
                raise _DamagedCrop(f'Cached crop checksum differs at {path}')
            self.crops_checked[source_digest] = stat
        return self._normalize(pixels)

    @staticmethod
    def _normalize(pixels):
        values = np.frombuffer(pixels, dtype=np.uint8).reshape(SHAPE).astype(np.float32)
        values /= np.float32(127.5)
        values -= np.float32(1.)
        return torch.from_numpy(values)

    def _pixels(self, index):
        row = self.rows[index]
        digest = bytes.fromhex(row['sha256'])
        if len(digest) != 32:
            raise ValueError('Expected a registered SHA-256 for the training photo')
        raw = None
        if self.sources_checked.get(index) != _stamp(Path(row['path']).stat()):
            raw = self._source_bytes(index, row)
        path = self.directory / digest.hex()[:2] / (digest.hex() + '.rgb')
        try:
            return self._read(path, digest)
        except (FileNotFoundError, _DamagedCrop):
            pass

        path.parent.mkdir(parents=True, exist_ok=True)
        with (path.parent / '.build.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return self._read(path, digest)
            except (FileNotFoundError, _DamagedCrop):
                pass
            if raw is None:
                raw = self._source_bytes(index, row)
            with Image.open(io.BytesIO(raw)) as image:
                result = rgb.center_crop(image)
            # The pinned transform maps each uint8 to v/127.5-1 as float32;
            # inversion by nearest integer recovers every original channel value.
            pixels = np.rint((result.numpy() + np.float32(1.)) * np.float32(127.5)).astype(np.uint8)
            encoded = pixels.tobytes(order='C')
            with atomic_path(path) as tmp, tmp.open('wb') as out:
                out.write(digest + hashlib.sha256(encoded).digest())
                out.write(encoded)
            return result

    def image(self, index, seed):
        value = self._pixels(index)
        return value.flip(-1) if _seed(seed, 'flip') % 2 else value


def install_source_cache(backend, root):
    """Install once on a newly built native backend; root is the scientific output directory."""
    if backend.native.kind not in ('guided_diffusion', 'sit_b_2'):
        return
    if backend.photos.domain not in ('imagenet', 'caltech256'):
        raise ValueError('Expected a registered ImageNet or Caltech256 training source')
    root = Path(root).resolve()
    store = SourcePixels(backend.photos, root.with_name(root.name + '.parallel') / 'source_pixels')
    backend.photos.image = store.image
