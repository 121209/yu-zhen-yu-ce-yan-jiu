#!/usr/bin/env python3
"""Reassemble payloads/parts/*.b64 into the archive tarball and extract it.

Used when this repository was filled through a text-only GitHub interface.
After extraction, run scripts/restore_binaries.py to write official .pt / .npy.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import tarfile
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unpack(root: Path) -> int:
    root = root.resolve()
    manifest_path = root / "payloads" / "MANIFEST.json"
    parts_dir = root / "payloads" / "parts"
    if not manifest_path.is_file():
        print(f"no payloads manifest at {manifest_path}; nothing to unpack")
        return 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blobs = []
    for rec in manifest["parts"]:
        path = parts_dir / rec["name"]
        if not path.is_file():
            print(f"missing payload part {path}", file=sys.stderr)
            return 1
        raw = base64.b64decode(path.read_text(encoding="ascii"), validate=False)
        if sha256_bytes(raw) != rec["raw_sha256"]:
            print(f"hash mismatch for {path}", file=sys.stderr)
            return 1
        if len(raw) != rec["raw_size"]:
            print(f"size mismatch for {path}", file=sys.stderr)
            return 1
        blobs.append(raw)
    tarball = b"".join(blobs)
    if sha256_bytes(tarball) != manifest["tarball_sha256"]:
        print("reassembled tarball hash mismatch", file=sys.stderr)
        return 1
    if len(tarball) != manifest["tarball_size"]:
        print("reassembled tarball size mismatch", file=sys.stderr)
        return 1
    tmp = root / "payloads" / "yu-zhen-archive-text.tar.gz"
    tmp.write_bytes(tarball)
    with tarfile.open(tmp, "r:gz") as tf:
        # Python 3.12+ has filter=; be conservative
        try:
            tf.extractall(root, filter="data")
        except TypeError:
            tf.extractall(root)
    tmp.unlink()
    print(f"extracted tarball ({len(tarball)} bytes) into {root}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Unpack text-safe archive payloads")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = p.parse_args()
    return unpack(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
