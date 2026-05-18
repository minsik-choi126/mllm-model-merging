"""Worker-safe LMDB reader used by all image-backed datasets.

LMDB envs cannot be forked; opening in __init__ and then letting torch DataLoader
fork workers will corrupt cursors. We therefore open the env lazily on first
access inside each worker process.
"""
from __future__ import annotations

import os
from pathlib import Path

import lmdb

from training.data.lmdb_builder import image_key


class LMDBReader:
    def __init__(self, lmdb_path: str | Path):
        self._path = str(Path(lmdb_path))
        self._env = None  # lazy

    def _open(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                self._path,
                subdir=os.path.isdir(self._path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1024,
            )
        return self._env

    def get(self, rel_path: str) -> bytes:
        env = self._open()
        with env.begin(write=False, buffers=True) as txn:
            blob = txn.get(image_key(rel_path))
        if blob is None:
            raise KeyError(f"{rel_path!r} not found in {self._path}")
        return bytes(blob)

    # Picklable: reset env on copy so workers re-open lazily.
    def __getstate__(self) -> dict:
        return {"_path": self._path, "_env": None}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
