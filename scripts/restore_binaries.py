#!/usr/bin/env python3
"""Decode official .pt / .npy payloads stored as *.b64 (text-safe Git objects).

The GitHub MCP used to fill this archive can only push text, so the official
binary snapshots are committed as base64 siblings (foo.pt.b64). This script
writes the original bytes next to them and checks sha256 against
results/MANIFEST.sha256.csv when that file is present.

Idempotent: already-correct binaries are left untouched.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from pathlib import Path

SUFFIXES = (".pt.b64", ".npy.b64")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(root: Path) -> dict[str, str]:
    path = root / "results" / "MANIFEST.sha256.csv"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        header = fh.readline()
        for line in fh:
            line = line.strip()
            if not line or line.startswith("sha256,"):
                continue
            sha, _size, rel = line.split(",", 2)
            out[rel] = sha
    return out


def restore(root: Path, check: bool = True) -> int:
    root = root.resolve()
    manifest = load_manifest(root) if check else {}
    payloads = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.name.endswith(SUFFIXES)
    )
    if not payloads:
        print(f"no *.pt.b64 / *.npy.b64 payloads under {root}")
        return 0
    errors = 0
    written = 0
    skipped = 0
    for payload in payloads:
        dest = Path(str(payload)[: -len(".b64")])
        data = base64.b64decode(payload.read_text(encoding="ascii"), validate=False)
        rel = dest.relative_to(root).as_posix()
        expected = manifest.get(rel)
        if expected and sha256_bytes(data) != expected:
            print(f"HASH MISMATCH in payload {payload.relative_to(root)}", file=sys.stderr)
            errors += 1
            continue
        if dest.is_file() and sha256_file(dest) == sha256_bytes(data):
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written += 1
        print(f"restored {rel} ({len(data)} bytes)")
    print(f"restore finished: wrote {written}, already ok {skipped}, errors {errors}")
    return 0 if errors == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Restore official .pt / .npy from base64 payloads")
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="archive repository root",
    )
    p.add_argument("--no-check", action="store_true", help="do not verify sha256 against MANIFEST")
    args = p.parse_args()
    return restore(args.root, check=not args.no_check)


if __name__ == "__main__":
    raise SystemExit(main())
