"""Offline LMDB builder for image shards.

Usage:
    python -m merit.data.lmdb_builder \\
        --src /data/vision_flan/images \\
        --out /data/vision_flan/images.lmdb \\
        --workers 8

This walks ``src`` recursively, hashes each image path, and stores the raw
image bytes under a SHA1 key. Downstream code (``merit.data.lmdb_dataset``)
looks up keys by the same hash.

LMDB environments are **not fork-safe**; you MUST open the env lazily per
DataLoader worker, which ``lmdb_dataset.LMDBReader`` handles for you.
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lmdb
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def image_key(relative_path: str) -> bytes:
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest().encode("ascii")


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def _iter_images(src: Path) -> list[Path]:
    return [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def build(src: Path, out: Path, map_size_gb: float = 256.0, workers: int = 4) -> int:
    src = src.resolve()
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    paths = _iter_images(src)
    if not paths:
        raise RuntimeError(f"no images under {src}")

    env = lmdb.open(
        str(out),
        map_size=int(map_size_gb * (1024**3)),
        subdir=True,
        readonly=False,
        lock=True,
        meminit=False,
        writemap=True,
    )
    written = 0
    txn = env.begin(write=True)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            fut_to_path = {pool.submit(_read_bytes, p): p for p in paths}
            for fut in tqdm(as_completed(fut_to_path), total=len(paths), desc="lmdb"):
                p = fut_to_path[fut]
                blob = fut.result()
                rel = p.relative_to(src).as_posix()
                txn.put(image_key(rel), blob)
                written += 1
                if written % 5000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
        txn.commit()
    finally:
        env.sync()
        env.close()
    return written


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--map-size-gb", type=float, default=256.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    n = build(args.src, args.out, map_size_gb=args.map_size_gb, workers=args.workers)
    print(f"[lmdb_builder] wrote {n} images → {args.out}")


if __name__ == "__main__":
    _cli()
