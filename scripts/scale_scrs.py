#!/usr/bin/env python3
"""
scale_scrs.py

Utility for debugging late CRS SCRS files.

Examples:
  # turn every valid scale into 1.0 (identity / no-op)
  python3 scripts/scale_scrs.py in.scrs out_identity.scrs --identity

  # blend scales toward 1.0 (25% of the original effect)
  python3 scripts/scale_scrs.py in.scrs out_blend.scrs --mix 0.25
"""

import argparse
import struct
from pathlib import Path

import numpy as np


MAGIC_SCRS = 0x53435253  # "SCRS"


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite scale values in a late CRS SCRS file")
    ap.add_argument("input", help="Input .scrs path")
    ap.add_argument("output", help="Output .scrs path")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--identity", action="store_true", help="Set every valid scale to 1.0")
    g.add_argument("--mix", type=float, help="Blend scales toward 1.0: out = 1 + mix * (in - 1)")
    args = ap.parse_args()

    data = Path(args.input).read_bytes()
    if len(data) < 24:
        raise ValueError("SCRS file too small")

    magic, version, n_layers, n_heads, n_dims, top_k = struct.unpack_from("<6I", data, 0)
    if magic != MAGIC_SCRS:
        raise ValueError(f"invalid SCRS magic: {magic:#x}")

    header_size = 24
    entry_size = top_k * 4 + top_k * 4
    expected = header_size + n_layers * n_heads * entry_size
    if len(data) != expected:
        raise ValueError(f"unexpected SCRS size: got {len(data)}, expected {expected}")

    out = bytearray(data)
    off = header_size

    changed = 0
    for _layer in range(n_layers):
        for _head in range(n_heads):
            idx = np.frombuffer(data, dtype=np.int32, count=top_k, offset=off).copy()
            off += top_k * 4
            scl = np.frombuffer(data, dtype=np.float32, count=top_k, offset=off).copy()

            valid = idx >= 0
            if args.identity:
                scl[valid] = 1.0
            else:
                scl[valid] = 1.0 + args.mix * (scl[valid] - 1.0)
            changed += int(valid.sum())

            out[off:off + top_k * 4] = scl.astype(np.float32).tobytes()
            off += top_k * 4

    Path(args.output).write_bytes(out)
    mode = "identity" if args.identity else f"mix={args.mix}"
    print(f"saved {args.output} ({mode}, changed {changed} valid scales)")


if __name__ == "__main__":
    main()
