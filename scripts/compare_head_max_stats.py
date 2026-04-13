#!/usr/bin/env python3
"""
Compare token-wise head-level absmax statistics across multiple ROPV dumps.

This directly checks whether rotation / RPN actually reduce the head block
scale driver for q4_0_head:

  head_absmax(t) = max_d |x_t[d]|

Statistics per (layer, head):
  mean, p50, p90, p99, p99.9, max

If multiple inputs are provided, the first one is treated as the reference and
reduction ratios against it are reported.
"""

import argparse
import csv
import os
import struct

import numpy as np


MAGIC_ROPV = 0x524F5056  # "ROPV"


def parse_named_input(text):
    if "=" not in text:
        raise ValueError(f"Expected NAME=PATH, got: {text}")
    name, path = text.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Invalid NAME=PATH: {text}")
    return name, path


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
            f"ROPV[{os.path.basename(path)}]: version={version}, layers={n_layers}, "
            f"heads={n_heads}, dims={n_dims}, tokens={n_tokens}"
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


def head_absmax_stats(arr):
    # arr: [T, D]
    token_max = np.max(np.abs(arr), axis=1)
    return {
        "mean": float(np.mean(token_max)),
        "p50": float(np.percentile(token_max, 50.0)),
        "p90": float(np.percentile(token_max, 90.0)),
        "p99": float(np.percentile(token_max, 99.0)),
        "p99_9": float(np.percentile(token_max, 99.9)),
        "max": float(np.max(token_max)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare token-wise head absmax stats across multiple ROPV files")
    parser.add_argument("--input", action="append", required=True,
                        help="Named input in the form NAME=/path/to/rope_values.bin (repeatable)")
    parser.add_argument("--post-rope", action="store_true",
                        help="Use post-RoPE values instead of pre-RoPE")
    parser.add_argument("--max-tokens", type=int, default=20480, help="Max tokens to load per layer")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    named_inputs = [parse_named_input(x) for x in args.input]
    ref_name = named_inputs[0][0]
    stage_name = "post" if args.post_rope else "pre"

    rows = []
    ref_stats = {}

    for input_idx, (name, path) in enumerate(named_inputs):
        for layer, pre, post, loaded, n_heads, n_dims in iter_ropv_layers(path, args.max_tokens):
            data = post if args.post_rope else pre
            if data is None or loaded <= 0:
                continue

            for head in range(n_heads):
                stats = head_absmax_stats(data[:, head, :])
                row = {
                    "model": name,
                    "stage": stage_name,
                    "layer": layer,
                    "head": head,
                    "tokens": int(loaded),
                    **stats,
                }
                rows.append(row)

                key = (layer, head)
                if input_idx == 0:
                    ref_stats[key] = stats

    rows.sort(key=lambda r: (r["layer"], r["head"], r["model"]))

    # Add reductions relative to reference.
    for row in rows:
        base = ref_stats.get((row["layer"], row["head"]))
        if base is None:
            continue
        for metric in ("mean", "p50", "p90", "p99", "p99_9", "max"):
            base_v = base[metric]
            cur_v = row[metric]
            delta = base_v - cur_v
            gain_pct = 100.0 * delta / max(base_v, 1e-12)
            row[f"{metric}_delta_vs_{ref_name}"] = delta
            row[f"{metric}_gain_pct_vs_{ref_name}"] = gain_pct

    csv_path = os.path.join(args.output_dir, f"head_absmax_stats_{stage_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {csv_path}")

    # Print a compact summary for non-reference models.
    for name, _ in named_inputs[1:]:
        subset = [r for r in rows if r["model"] == name]
        if not subset:
            continue
        mean_p99_gain = np.mean([r[f"p99_gain_pct_vs_{ref_name}"] for r in subset])
        mean_p999_gain = np.mean([r[f"p99_9_gain_pct_vs_{ref_name}"] for r in subset])
        mean_max_gain = np.mean([r[f"max_gain_pct_vs_{ref_name}"] for r in subset])
        print(
            f"  {name} vs {ref_name} ({stage_name}): "
            f"mean p99 gain={mean_p99_gain:.3f}% | "
            f"mean p99.9 gain={mean_p999_gain:.3f}% | "
            f"mean max gain={mean_max_gain:.3f}%"
        )


if __name__ == "__main__":
    main()
