#!/usr/bin/env python3
"""
Analyze post-RoPE pair-norm tails and recommend clipping actions.

This script reads a raw ROPV dump, computes per-(layer, head, pair) pair-norm
statistics on post-RoPE values,

    r_t = sqrt(x_t^2 + y_t^2)

and uses max/q99.9 as a simple tail-severity score.

By default:
- q99.9 is used as the recommended clipping threshold
- shape CSV is optional
- elongated pairs are not excluded from clipping; they are tagged as
  "clip_and_check_rotation" style actions when the tail is large
"""

import argparse
import csv
import os
import struct

import numpy as np

MAGIC_ROPV = 0x524F5056  # "ROPV"


def read_chunk(f, n_floats):
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        print(f"  [warn] Expected {n_floats} floats, got {got}")
    return np.frombuffer(raw[:got * 4], dtype=np.float32).copy()


def read_rope_values(file_path, max_tokens=10000):
    with open(file_path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]
        n_tokens = struct.unpack("I", f.read(4))[0]
        stride = n_heads * n_dims

        print(f"File: {file_path}")
        print(f"  Version={version}, Layers={n_layers}, Heads={n_heads}, Dims={n_dims}, Tokens={n_tokens}")

        layer_data = []
        for layer in range(n_layers):
            hdr = f.read(4)
            if len(hdr) < 4:
                print(f"  [warn] EOF at layer {layer} pre_count")
                layer_data.append({"pre": None, "post": None})
                continue
            pre_count = struct.unpack("I", hdr)[0]

            pre_rope = None
            if pre_count > 0:
                actual_tokens_pre = pre_count // stride
                use_pre = min(actual_tokens_pre, max_tokens)
                read_floats = use_pre * stride
                skip_floats = pre_count - read_floats
                pre_rope = read_chunk(f, read_floats)
                if skip_floats > 0:
                    f.seek(skip_floats * 4, 1)
                if len(pre_rope) == read_floats:
                    pre_rope = pre_rope.reshape(use_pre, n_heads, n_dims)
                else:
                    pre_rope = None

            hdr = f.read(4)
            if len(hdr) < 4:
                print(f"  [warn] EOF at layer {layer} post_count")
                layer_data.append({"pre": pre_rope, "post": None})
                continue
            post_count = struct.unpack("I", hdr)[0]

            post_rope = None
            if post_count > 0:
                actual_tokens_post = post_count // stride
                use_post = min(actual_tokens_post, max_tokens)
                read_floats = use_post * stride
                skip_floats = post_count - read_floats
                post_rope = read_chunk(f, read_floats)
                if skip_floats > 0:
                    f.seek(skip_floats * 4, 1)
                if len(post_rope) == read_floats:
                    post_rope = post_rope.reshape(use_post, n_heads, n_dims)
                else:
                    post_rope = None

            layer_data.append({"pre": pre_rope, "post": post_rope})

            if layer == 0:
                pre_tok = pre_rope.shape[0] if pre_rope is not None else 0
                post_tok = post_rope.shape[0] if post_rope is not None else 0
                print(
                    f"  Layer 0: pre_count={pre_count} -> {pre_tok} tokens loaded, "
                    f"post_count={post_count} -> {post_tok} tokens loaded"
                )
                print(f"  (max_tokens={max_tokens})")

        return {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_dims": n_dims,
            "n_tokens": n_tokens,
            "layers": layer_data,
        }


def compute_rope_freqs(n_dims, freq_base=500000.0):
    n_pairs = n_dims // 2
    return np.array([freq_base ** (-2.0 * i / n_dims) for i in range(n_pairs)], dtype=np.float64)


def synthesize_post_if_missing(data, freq_base):
    has_post = any(layer["post"] is not None for layer in data["layers"])
    if has_post:
        return "raw_post"

    print("\n[warn] No post-RoPE data found. Synthesizing post values from pre-RoPE.")
    freqs = compute_rope_freqs(data["n_dims"], freq_base)
    for layer_idx in range(data["n_layers"]):
        pre = data["layers"][layer_idx]["pre"]
        if pre is None:
            continue

        n_tok = pre.shape[0]
        post = np.zeros_like(pre)
        for pair_i in range(data["n_dims"] // 2):
            d0 = pair_i * 2
            d1 = d0 + 1
            for t in range(n_tok):
                theta = t * freqs[pair_i]
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                post[t, :, d0] = pre[t, :, d0] * cos_t - pre[t, :, d1] * sin_t
                post[t, :, d1] = pre[t, :, d0] * sin_t + pre[t, :, d1] * cos_t
        data["layers"][layer_idx]["post"] = post

    return "synth_post"


def load_shape_rows(csv_path):
    shape_rows = {}
    if csv_path is None:
        return shape_rows

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["layer"]), int(row["head"]), int(row["pair"]))
            shape_rows[key] = row

    print(f"Loaded shape CSV: {csv_path} ({len(shape_rows)} rows)")
    return shape_rows


def classify_tail_label(ratio, mild_ratio, clip_ratio, strong_ratio):
    if ratio < mild_ratio:
        return "no_clip"
    if ratio < clip_ratio:
        return "mild_clip"
    if ratio < strong_ratio:
        return "clip_recommended"
    return "strong_clip"


def classify_shape_aware_action(shape_label, tail_label):
    elongated = shape_label == "elongated"

    if tail_label == "no_clip":
        if elongated:
            return "rotation_check"
        return "no_clip"

    if tail_label == "mild_clip":
        if elongated:
            return "mild_clip_check_rotation"
        return "mild_clip"

    if tail_label == "clip_recommended":
        if elongated:
            return "clip_and_check_rotation"
        return "clip_recommended"

    if elongated:
        return "strong_clip_and_check_rotation"
    return "strong_clip"


def analyze_clip_candidates(
    data,
    layers,
    heads,
    pair_start,
    pair_end,
    shape_rows,
    source_kind,
    mild_ratio,
    clip_ratio,
    strong_ratio,
):
    rows = []
    n_pairs_total = data["n_dims"] // 2
    if pair_end is None:
        pair_end = n_pairs_total
    pair_end = min(pair_end, n_pairs_total)

    for layer_idx in layers:
        post = data["layers"][layer_idx]["post"]
        if post is None:
            continue

        for head_idx in heads:
            for pair_idx in range(pair_start, pair_end):
                d0 = pair_idx * 2
                d1 = d0 + 1

                x = post[:, head_idx, d0].astype(np.float64)
                y = post[:, head_idx, d1].astype(np.float64)
                r = np.sqrt(x * x + y * y)

                if r.size == 0:
                    continue

                q99 = float(np.percentile(r, 99.0))
                q995 = float(np.percentile(r, 99.5))
                q999 = float(np.percentile(r, 99.9))
                rmax = float(np.max(r))
                rmean = float(np.mean(r))
                rmedian = float(np.median(r))
                eps = 1e-12
                ratio_max_q999 = rmax / max(q999, eps)
                ratio_max_q995 = rmax / max(q995, eps)
                ratio_max_q99 = rmax / max(q99, eps)
                reduction_frac = 0.0 if rmax <= eps else 1.0 - (q999 / rmax)
                clipped_fraction = float(np.mean(r > q999))

                shape_row = shape_rows.get((layer_idx, head_idx, pair_idx), {})
                shape_label = shape_row.get("shape_label", "unknown")
                orientation_label = shape_row.get("orientation_label", "unknown")

                tail_label = classify_tail_label(
                    ratio_max_q999,
                    mild_ratio=mild_ratio,
                    clip_ratio=clip_ratio,
                    strong_ratio=strong_ratio,
                )
                action = classify_shape_aware_action(shape_label, tail_label)

                rows.append(
                    {
                        "layer": layer_idx,
                        "head": head_idx,
                        "pair": pair_idx,
                        "dim0": d0,
                        "dim1": d1,
                        "source": source_kind,
                        "n_tokens_used": int(r.size),
                        "shape_label": shape_label,
                        "orientation_label": orientation_label,
                        "r_mean": rmean,
                        "r_median": rmedian,
                        "q99": q99,
                        "q99_5": q995,
                        "q99_9": q999,
                        "r_max": rmax,
                        "ratio_max_q99": ratio_max_q99,
                        "ratio_max_q99_5": ratio_max_q995,
                        "ratio_max_q99_9": ratio_max_q999,
                        "clip_tau": q999,
                        "clip_reduction_frac": reduction_frac,
                        "clip_fraction_at_tau": clipped_fraction,
                        "tail_label": tail_label,
                        "shape_aware_action": action,
                    }
                )

    rows.sort(key=lambda row: (-row["ratio_max_q99_9"], row["layer"], row["head"], row["pair"]))
    return rows


def save_rows(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "clip_candidates.csv")
    if not rows:
        print("  [warn] No rows to save.")
        return csv_path

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def print_summary(rows, topk=15):
    if not rows:
        print("\nNo candidate rows found.")
        return

    tail_counts = {}
    action_counts = {}
    for row in rows:
        tail_counts[row["tail_label"]] = tail_counts.get(row["tail_label"], 0) + 1
        action_counts[row["shape_aware_action"]] = action_counts.get(row["shape_aware_action"], 0) + 1

    print("\nTail severity counts:")
    for key in sorted(tail_counts.keys()):
        print(f"  {key}: {tail_counts[key]}")

    print("\nShape-aware action counts:")
    for key in sorted(action_counts.keys()):
        print(f"  {key}: {action_counts[key]}")

    print(f"\nTop {min(topk, len(rows))} pairs by max/q99.9:")
    for row in rows[:topk]:
        print(
            f"  L{row['layer']:02d} H{row['head']:02d} P{row['pair']:02d} | "
            f"shape={row['shape_label']} | action={row['shape_aware_action']} | "
            f"ratio={row['ratio_max_q99_9']:.3f} | "
            f"q99.9={row['q99_9']:.5f} | max={row['r_max']:.5f} | "
            f"reduction={100.0 * row['clip_reduction_frac']:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze post-RoPE pair-norm tails and clipping candidates")
    parser.add_argument("rope_file", help="Path to rope_values.bin / ROPV file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for clip_candidates.csv")
    parser.add_argument("--shape-csv", type=str, default=None, help="Optional shape_orientation_table.csv")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Maximum tokens to load per layer")
    parser.add_argument("--freq-base", type=float, default=500000.0, help="Used only if post values must be synthesized")
    parser.add_argument("--pair-start", type=int, default=32, help="First pair to analyze")
    parser.add_argument("--pair-end", type=int, default=None, help="Exclusive pair end")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Layer indices to analyze")
    parser.add_argument("--heads", type=int, nargs="*", default=None, help="Head indices to analyze")
    parser.add_argument("--mild-ratio", type=float, default=1.2, help="max/q99.9 threshold for mild clip")
    parser.add_argument("--clip-ratio", type=float, default=1.5, help="max/q99.9 threshold for clip recommendation")
    parser.add_argument("--strong-ratio", type=float, default=2.0, help="max/q99.9 threshold for strong clip")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.rope_file), "clip_analysis")

    print("=" * 60)
    print("RoPE Pair-Norm Tail / Clip Candidate Analysis")
    print("=" * 60)

    print(f"\nLoading data (max {args.max_tokens} tokens)...")
    data = read_rope_values(args.rope_file, max_tokens=args.max_tokens)
    source_kind = synthesize_post_if_missing(data, args.freq_base)

    layers = args.layers if args.layers is not None else list(range(data["n_layers"]))
    heads = args.heads if args.heads is not None else list(range(data["n_heads"]))

    shape_rows = load_shape_rows(args.shape_csv)

    print("\nComputing pair-norm tail statistics...")
    rows = analyze_clip_candidates(
        data=data,
        layers=layers,
        heads=heads,
        pair_start=args.pair_start,
        pair_end=args.pair_end,
        shape_rows=shape_rows,
        source_kind=source_kind,
        mild_ratio=args.mild_ratio,
        clip_ratio=args.clip_ratio,
        strong_ratio=args.strong_ratio,
    )

    csv_path = save_rows(rows, args.output_dir)
    print_summary(rows)

    print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()
