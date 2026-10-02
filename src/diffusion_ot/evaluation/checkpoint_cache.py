"""One-process, one-checkpoint cache for controlled evaluation sweeps."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_identity(path):
    if path is None:
        return None
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class CheckpointEvaluationCache:
    """Never shares tensors across checkpoint/config/weight/device identities.

    Objects are read-only to callers. No persistent cache is trusted: every new
    process loads checkpoint weights and encodes the selected data once again.
    """
    def __init__(self):
        self.identity = None
        self.values = {}
        self.hits = Counter()
        self.misses = Counter()

    def bind(self, identity):
        identity = fingerprint(identity)
        if self.identity != identity:
            self.clear()
            self.identity = identity

    def clear(self):
        self.values.clear()
        self.identity = None

    def get_or_create(self, kind, key, factory):
        cache_key = (kind, fingerprint(key))
        if cache_key not in self.values:
            self.misses[kind] += 1
            self.values[cache_key] = factory()
        else:
            self.hits[kind] += 1
        return self.values[cache_key]


def cached(cache, kind, key, factory):
    return factory() if cache is None else cache.get_or_create(kind, key, factory)
