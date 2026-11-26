"""Exercise cached native source crops without loading a model or touching campaign checkpoints."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from experiment_2.code.generation_backend import _RealTrainingPhotos
from experiment_2.code.input_paths import read_registered_json
from experiment_2.parallel_strength import source_pixels as cache
from runtime.comparison.protocol_v2.generative_reference import _seed
from runtime.comparison.protocol_v2.storage import atomic_path


def backend_for(rows, domain='imagenet', kind='guided_diffusion'):
    photos = _RealTrainingPhotos.__new__(_RealTrainingPhotos)
    photos.rows, photos.domain, photos.checked_files = rows, domain, set()
    photos.order, photos.order_key = None, None
    return SimpleNamespace(photos=photos, native=SimpleNamespace(kind=kind))


class SourcePixelTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.root = self.base / 'campaign'
        data = np.random.default_rng(10).integers(0, 256, (560, 777, 3), dtype=np.uint8)
        self.photo = self.base / 'source.png'
        Image.fromarray(data).save(self.photo)
        data = self.photo.read_bytes()
        self.row = {'path': str(self.photo), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    def install(self, kind='guided_diffusion'):
        backend = backend_for([self.row], kind=kind)
        cache.install_source_cache(backend, self.root)
        return backend.photos

    def cache_path(self):
        digest = self.row['sha256']
        return self.base / 'campaign.parallel/source_pixels' / digest[:2] / (digest + '.rgb')

    def test_all_uint8_values_and_both_flips_match_exactly(self):
        ramp = np.broadcast_to(np.arange(256, dtype=np.uint8), cache.SHAPE)
        got = cache.SourcePixels._normalize(ramp.tobytes())
        self.assertTrue(torch.equal(got, torch.from_numpy(ramp.astype(np.float32) / 127.5 - 1)))
        native = backend_for([self.row]).photos
        seeds = [next(s for s in range(20) if _seed(s, 'flip') % 2 == sign) for sign in (0, 1)]
        expected = [native.image(0, s) for s in seeds]
        first = self.install()
        self.assertTrue(torch.equal(first.image(0, seeds[0]), expected[0]))
        self.assertFalse(self.root.exists())
        self.assertEqual(self.cache_path().stat().st_size, cache.HEADER_BYTES + cache.PIXEL_BYTES)
        second = self.install('sit_b_2')
        with patch.object(cache.rgb, 'center_crop', side_effect=AssertionError('duplicate decoding')):
            for seed, ref in zip(seeds, expected):
                self.assertTrue(torch.equal(first.image(0, seed), ref))
                self.assertTrue(torch.equal(second.image(0, seed), ref))

    def test_source_change_is_rejected_and_damaged_pixels_are_rebuilt(self):
        photos = self.install()
        expected = photos.image(0, 1)
        path = self.cache_path()
        for damage in ('truncation', 'pixel'):
            with path.open('r+b') as f:
                if damage == 'truncation':
                    f.truncate(53)
                else:
                    f.seek(-1, 2)
                    changed = f.read(1)[0] ^ 0xff
                    f.seek(-1, 2)
                    f.write(bytes([changed]))
            self.assertTrue(torch.equal(photos.image(0, 1), expected))
        raw = self.photo.read_bytes()
        self.photo.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xff]))
        with self.assertRaisesRegex(ValueError, 'Registered training photo changed'):
            photos.image(0, 1)
        with self.assertRaisesRegex(ValueError, 'Registered training photo changed'):
            self.install().image(0, 1)

    def test_concurrent_builds_share_one_crop(self):
        photos = [self.install() for _ in range(4)]
        ready = threading.Barrier(len(photos))
        def get(backend):
            ready.wait(timeout=15)
            return backend.image(0, 2)
        with patch.object(cache.rgb, 'center_crop', wraps=cache.rgb.center_crop) as crop:
            with ThreadPoolExecutor(max_workers=len(photos)) as pool:
                results = list(pool.map(get, photos))
            self.assertEqual(crop.call_count, 1)
        self.assertTrue(all(torch.equal(results[0], value) for value in results))

    def test_interrupted_write_never_publishes_a_partial_crop(self):
        photos = self.install()
        @contextmanager
        def interrupted(path):
            with atomic_path(path) as tmp:
                yield tmp
                raise InterruptedError('before atomic rename')
        with patch.object(cache, 'atomic_path', interrupted):
            with self.assertRaises(InterruptedError):
                photos.image(0, 3)
        self.assertFalse(self.cache_path().exists())
        native = backend_for([self.row]).photos
        self.assertTrue(torch.equal(self.install().image(0, 3), native.image(0, 3)))

    def test_real_registered_batches(self):
        cfg = Path(__file__).resolve().parents[1] / 'config'
        design = read_registered_json(cfg / 'NEXT_EXPERIMENT_DESIGN.json')
        seed = json.loads((cfg / 'STRENGTH_TRAINING.json').read_text())['unet']['seed']
        for domain in ('imagenet', 'caltech256'):
            manifest = design['embedding']['native_generator_training']['train_pools'][domain]['manifest']
            if not Path(manifest['path']).is_file():
                self.skipTest('registered training data are not mounted')
            rows = read_registered_json(manifest['path'])
            backend = backend_for(rows, domain)
            indices = backend.photos.batch_indices(0, 32, seed)[:8]
            draws = [(i, _seed(seed, 'source_training', 0, j)) for j, i in enumerate(indices)]
            refs = [backend.photos.image(*draw) for draw in draws]
            cache.install_source_cache(backend, self.root)
            self.assertTrue(all(torch.equal(backend.photos.image(*draw), ref) for draw, ref in zip(draws, refs)))
            same_domain_sit = backend_for(rows, domain, 'sit_b_2')
            cache.install_source_cache(same_domain_sit, self.root)
            with patch.object(cache.rgb, 'center_crop', side_effect=AssertionError('duplicate native crop')):
                for draw, ref in zip(draws, refs):
                    self.assertTrue(torch.equal(same_domain_sit.photos.image(*draw), ref))


if __name__ == '__main__':
    unittest.main()
