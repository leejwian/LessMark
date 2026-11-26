"""Codebook construction and truth-free decisions; scoring lives elsewhere."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Decision:
    status: str
    recipient: int | None = None
    recipients: tuple[int, ...] = ()
    details: dict = field(default_factory=dict)


def gf4_mul(a, b):
    # GF(2)[x]/(x^2+x+1); integers encode polynomial coefficients.
    out = 0
    while b:
        if b & 1:
            out ^= a
        b >>= 1
        a <<= 1
        if a & 4:
            a ^= 7
    return out


def bibd256():
    points = list(itertools.product(range(4), repeat=4))
    index = {p: i for i, p in enumerate(points)}
    directions = [v for v in points[1:] if next(x for x in v if x) == 1]
    blocks = set()
    for v in directions:
        for p in points:
            blocks.add(tuple(sorted(index[tuple(a ^ gf4_mul(t, b) for a, b in zip(p, v))] for t in range(4))))
    blocks = np.asarray(sorted(blocks), dtype=np.int64)
    codes = np.ones((5440, 256), dtype=np.uint8)  # API rows = IDs; persisted transpose documented.
    codes[np.arange(5440)[:, None], blocks] = 0
    validate_bibd(codes)
    return codes, blocks


def validate_codes(codes):
    c = np.asarray(codes)
    if c.ndim != 2 or len(c) < 1 or not np.isin(c, [0, 1]).all():
        raise ValueError("invalid binary registry")
    if len(np.unique(c, axis=0)) != len(c):
        raise ValueError("duplicate registered code: registry invalid")
    return c


def validate_bibd(codes):
    c = validate_codes(codes)
    if c.shape != (5440, 256) or not np.all((c == 0).sum(axis=1) == 4):
        raise ValueError("requires (256,4,1)-BIBD complement, M=5440")
    a = (1 - c).astype(np.int64)
    incidence = a.T @ a
    if not np.all(np.diag(incidence) == 85) or not np.all(incidence[~np.eye(256, dtype=bool)] == 1):
        raise ValueError("BIBD pair multiplicity invalid")


class Registry:
    def __init__(self, codes, bibd=False):
        self.codes = validate_codes(codes).copy()
        if bibd:
            validate_bibd(self.codes)
        self.bibd = bibd
        self.blocks = np.stack([np.flatnonzero(c == 0) for c in self.codes]) if bibd else None
        self.lookup = {c.tobytes(): i for i, c in enumerate(self.codes.astype(np.uint8))}

    def exact(self, bits):
        b = np.asarray(bits) if bits is not None else np.array([])
        if b.shape != self.codes.shape[1:] or not np.isin(b, [0, 1]).all():
            return Decision("invalid", details={"reason": "no valid complete binary vector"})
        r = self.lookup.get(b.astype(np.uint8).tobytes())
        return Decision("id" if r is not None else "no_match", recipient=r)

    def trace(self, bits):
        if not self.bibd:
            raise ValueError("AND tracing requires validated BIBD")
        b = np.asarray(bits) if bits is not None else np.array([])
        if b.shape != (256,) or not np.isin(b, [0, 1]).all():
            return Decision("extraction_error", details={"reason": "invalid aggregate code"})
        zero = b == 0
        r = np.flatnonzero(np.all(zero[self.blocks], axis=1))
        if 1 <= len(r) <= 3 and np.array_equal(np.unique(self.blocks[r]), np.flatnonzero(zero)):
            return Decision("set", recipients=tuple(map(int, r)))
        return Decision("empty_set")


def balanced_coalitions(query_ids, seed):
    """Four cyclic orbits per size: exact 128 distinct coalitions, degree 8/12."""
    if len(set(query_ids)) != 32:
        raise ValueError("representative design requires 32 distinct recipients")
    ids = np.random.default_rng(seed).permutation(query_ids).tolist()
    out = {}
    for size, patterns in [(2, [(0, d) for d in (1, 2, 3, 4)]), (3, [(0, 1, d) for d in (3, 5, 7, 9)])]:
        rows = sorted({tuple(sorted(ids[(i + d) % 32] for d in pat)) for pat in patterns for i in range(32)})
        assert len(rows) == 128
        assert all(sum(x in row for row in rows) == size * 4 for x in ids)
        out[str(size)] = rows
    return out


def random_registry(k, m, seed, bridge=None):
    if m > 2 ** k:
        raise ValueError("M exceeds code space")
    rng = np.random.default_rng(seed)
    if bridge is not None:
        base = validate_codes(bridge)
        if base.shape != (9, 8) or k % 8:
            raise ValueError("bridge requires original 9x8 codes and k divisible by 8")
        pieces = []
        for _ in range(k // 8):
            pieces.append(base[:, rng.permutation(8)] ^ rng.integers(0, 2, 8, dtype=np.uint8))
        return np.concatenate(pieces, axis=1)
    # Regeneration handles code collisions, never recovery outcomes.
    rows, seen = [], set()
    while len(rows) < m:
        c = rng.integers(0, 2, k, dtype=np.uint8)
        if c.tobytes() not in seen:
            seen.add(c.tobytes())
            rows.append(c)
    return np.stack(rows)
