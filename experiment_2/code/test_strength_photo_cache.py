"""Behavior of the shared FFHQ source-training RGB cache."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from experiment_2.code.generation_backend import _RealTrainingPhotos


class FFHQTrainingCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / '00000.png'
        pattern = np.random.default_rng(42).integers(0, 256, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(pattern).save(self.source)
        self.manifest = self.root / 'training_inputs.json'
        self.manifest.write_text('{"condition":"FFHQ cache test"}')
        entry = {'train_pool': 'ffhq'}
        pool = {'image_directory': str(self.root), 'official_training_ids_inclusive': [0, 0],
                'available_images': 1}
        self.design = {'embedding': {'native_generator_training': {'conditions': {'ffhq': entry},
                                                                   'train_pools': {'ffhq': pool}}}}

    def photos(self, *, register=True):
        photos = _RealTrainingPhotos(self.design, 'ffhq', {})
        if register:
            photos.register_ffhq_rgb_cache(self.manifest)
        return photos

    def test_shared_pixels_match_uncached_and_detect_mutation(self):
        original = self.photos(register=False).image(0, 5)
        photos = self.photos()
        torch.testing.assert_close(photos.image(0, 5), original, rtol=0, atol=0)
        cache, = photos.ffhq_rgb_cache.glob('*/*.pt')
        saved = torch.load(cache, weights_only=True)
        self.assertEqual(saved['source_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
        resumed = self.photos()
        with patch.object(Image, 'open', side_effect=AssertionError('duplicate decode')):
            self.assertTrue(torch.equal(photos.image(0, 8), original))
            self.assertTrue(torch.equal(resumed.image(0, 9), original))
            stat = self.source.stat()
            os.utime(self.source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            self.assertTrue(torch.equal(resumed.image(0, 5), original))
            self.source.write_bytes(b'changed source')
            with self.assertRaisesRegex(ValueError, 'source bytes changed'):
                resumed.image(0, 5)

    def test_partial_write_is_retryable_and_damaged_pixels_are_detected(self):
        photos = self.photos()
        def incomplete(record, stream):
            stream.write(b'partial')
            raise OSError('test interrupted write')
        with patch.object(torch, 'save', side_effect=incomplete), self.assertRaises(OSError):
            photos.image(0, 1)
        self.assertFalse(list(photos.ffhq_rgb_cache.glob('*/*.pt')))
        resumed = self.photos()
        resumed.image(0, 1)
        cache, = resumed.ffhq_rgb_cache.glob('*/*.pt')
        saved = torch.load(cache, weights_only=True)
        saved['pixels'][0, 0, 0] ^= 1
        torch.save(saved, cache)
        with self.assertRaisesRegex(ValueError, 'Damaged FFHQ'):
            resumed.image(0, 1)


if __name__ == '__main__':
    unittest.main()
