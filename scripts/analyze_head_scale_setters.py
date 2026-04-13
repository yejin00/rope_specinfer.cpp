#!/usr/bin/env python3
"""
Analyze which dimensions / pairs actually set the head-level absmax most often.

For each token, layer, head:
  setter_dim(t)  = argmax_d |x_t[d]|
  setter_pair(t) = setter_dim(t) // 2
  setter_block(t)= setter_dim(t) // 32

This helps explain whether early or late bands are actually dominating
the q4_0_head scale.
"""

import argparse
import csv
import json
import math
import os
import struct
from collections import Counter

import numpy as np


MAGIC_ROPV = 0x524F5056  # "ROPV"


def read_chunk_f32(f, n_floats):
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        raise ValueError(f"Expected {n_floats} float32 values, got {got}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def iter_ropv_layers(path, max_tokens):
    with open(path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]
        n_tokens = struct.unpack("I", f.read(4))[0]
        stride = n_heads * n_dims

        print(
            f"ROPV: version={version}, layers={n_layers}, heads={n_heads}, "
            f"dims={n_dims}, tokens={n_tokens}"
        )

        for layer in range(n_layers):
            pre_count = struct.unpack("I", f.read(4))[0]
            pre = None
            pre_loaded = 0
            if pre_count > 0:
                pre_tokens = pre_count // stride
                pre_loaded = min(pre_tokens, max_tokens)
                read_floats = pre_loaded * stride
                skip_floats = pre_count - read_floats
                pre = read_chunk_f32(f, read_floats).reshape(pre_loaded, n_heads, n_dims)
                if skip_floats > 0:
                    f.seek(skip_floats * 4, 1)

            post_count = struct.unpack("I", f.read(4))[0]
            post = None
            post_loaded = 0
            if post_count > 0:
                post_tokens = post_count // stride
                post_loaded = min(post_tokens, max_tokens)
                read_floats = post_loaded * stride
                skip_floats = post_count - read_floats
                post = read_chunk_f32(f, read_floats).reshape(post_loaded, n_heads, n_dims)
                if skip_floats > 0:
                    f.seek(skip_floats * 4, 1)

            loaded = min(pre_loaded, post_loaded) if post is not None else pre_loaded
            if pre is not None and pre.shape[0] != loaded:
                pre = pre[:loaded]
            if post is not None and post.shape[0] != loaded:
                post = post[:loaded]

            yield layer, pre, post, loaded, n_heads, n_dims


def top_counts(counter, k=8):
    return counter.most_common(k)


def fmt_top_counts(items):
    return ";".join(f"{idx}:{cnt}" for idx, cnt in items)


def main():
    parser = argparse.ArgumentParser(description="Analyze head-level scale setter dims/pairs from ROPV")
    parser.add_argument("input_ropv", help="Input rope_values*.bin (ROPV)")
    parser.add_argument("--post-rope", action="store_true",
                        help="Analyze post-RoPE values instead of pre-RoPE")
    parser.add_argument("--max-tokens", type=int, default=20480, help="Max tokens to load per layer")
    parser.add_argument("--pair-split", type=int, default=32,
                        help="Early/late split in pair index (default: 32)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--top-k", type=int, default=8, help="How many top dims/pairs to store per head")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    details = {}
    stage_name = "post" if args.post_rope else "pre"

    for layer, pre, post, loaded, n_heads, n_dims in iter_ropv_layers(args.input_ropv, args.max_tokens):
        data = post if args.post_rope else pre
        if data is None or loaded <= 0:
            continue

        n_pairs = n_dims // 2
        for head in range(n_heads):
            head_data = data[:, head, :]  # [T, D]
            setter_dims = np.abs(head_data).argmax(axis=1)
            setter_pairs = setter_dims // 2
            setter_blocks = setter_dims // 32

            dim_counter = Counter(int(x) for x in setter_dims.tolist())
            pair_counter = Counter(int(x) for x in setter_pairs.tolist())
            block_counter = Counter(int(x) for x in setter_blocks.tolist())

            top_dim = dim_counter.most_common(1)[0]
            top_pair = pair_counter.most_common(1)[0]

            early_tokens = int(np.sum(setter_pairs < args.pair_split))
            late_tokens = int(np.sum(setter_pairs >= args.pair_split))
            early_frac = early_tokens / loaded
            late_frac = late_tokens / loaded

            key = f"L{layer}_H{head}"
            details[key] = {
                "layer": layer,
                "head": head,
                "tokens": int(loaded),
                "stage": stage_name,
                "top_dims": [{"dim": idx, "count": cnt} for idx, cnt in top_counts(dim_counter, args.top_k)],
                "top_pairs": [{"pair": idx, "count": cnt} for idx, cnt in top_counts(pair_counter, args.top_k)],
                "block_counts": {str(idx): int(cnt) for idx, cnt in sorted(block_counter.items())},
                "early_pair_tokens": int(early_tokens),
                "late_pair_tokens": int(late_tokens),
                "early_pair_frac": float(early_frac),
                "late_pair_frac": float(late_frac),
            }

            summary_rows.append({
                "layer": layer,
                "head": head,
                "tokens": int(loaded),
                "stage": stage_name,
                "top_setter_dim": int(top_dim[0]),
                "top_setter_dim_freq": int(top_dim[1]),
                "top_setter_dim_frac_pct": 100.0 * float(top_dim[1]) / loaded,
                "top_setter_pair": int(top_pair[0]),
                "top_setter_pair_freq": int(top_pair[1]),
                "top_setter_pair_frac_pct": 100.0 * float(top_pair[1]) / loaded,
                "early_pair_tokens": int(early_tokens),
                "late_pair_tokens": int(late_tokens),
                "early_pair_frac_pct": 100.0 * early_frac,
                "late_pair_frac_pct": 100.0 * late_frac,
                "block0_frac_pct": 100.0 * block_counter.get(0, 0) / loaded,
                "block1_frac_pct": 100.0 * block_counter.get(1, 0) / loaded,
                "block2_frac_pct": 100.0 * block_counter.get(2, 0) / loaded,
                "block3_frac_pct": 100.0 * block_counter.get(3, 0) / loaded,
                "top_dims": fmt_top_counts(top_counts(dim_counter, args.top_k)),
                "top_pairs": fmt_top_counts(top_counts(pair_counter, args.top_k)),
            })

    summary_rows.sort(key=lambda r: (r["layer"], r["head"]))

    csv_path = os.path.join(args.output_dir, f"scale_setter_summary_{stage_name}.csv")
    json_path = os.path.join(args.output_dir, f"scale_setter_details_{stage_name}.json")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    overall_early = np.mean([r["early_pair_frac_pct"] for r in summary_rows]) if summary_rows else float("nan")
    overall_late = np.mean([r["late_pair_frac_pct"] for r in summary_rows]) if summary_rows else float("nan")
    top_early = sorted(summary_rows, key=lambda r: r["early_pair_frac_pct"], reverse=True)[:8]
    top_late = sorted(summary_rows, key=lambda r: r["late_pair_frac_pct"], reverse=True)[:8]

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"\nOverall setter split ({stage_name}): early<{args.pair_split} = {overall_early:.2f}% | late>={args.pair_split} = {overall_late:.2f}%")
    print("\nTop heads by early-pair setter fraction:")
    for row in top_early:
        print(
            f"  L{row['layer']:02d} H{row['head']:02d} | early={row['early_pair_frac_pct']:.2f}% "
            f"| late={row['late_pair_frac_pct']:.2f}% | top_pair={row['top_setter_pair']}"
        )
    print("\nTop heads by late-pair setter fraction:")
    for row in top_late:
        print(
            f"  L{row['layer']:02d} H{row['head']:02d} | late={row['late_pair_frac_pct']:.2f}% "
            f"| early={row['early_pair_frac_pct']:.2f}% | top_pair={row['top_setter_pair']}"
        )


if __name__ == "__main__":
    main()
