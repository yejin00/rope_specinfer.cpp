#!/usr/bin/env python3
"""
Filter a rotation CSV by simulated token-wise q4_0_head fake-quant error.

For each candidate (layer, head, pair, alpha) from the input CSV:
  1. Read post-RoPE K activations from a ROPV dump
  2. For each token, fake-quant/dequant the full 128-d head vector with q4_0_head
  3. Rotate only the candidate pair by alpha, fake-quant/dequant again
  4. Measure mean token-wise error reduction
  5. Keep only candidates with positive gain (or above a threshold)

This is more directly aligned with q4_0_head KV-cache quantization than
shape-based heuristics.
"""

import argparse
import csv
import math
import struct
from collections import defaultdict

import numpy as np


MAGIC_ROPV = 0x524F5056  # "ROPV"
QK4_0_HEAD = 128


def read_chunk_f32(f, n_floats):
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        raise ValueError(f"Expected {n_floats} float32 values, got {got}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def load_rotation_csv(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            row["_layer"] = int(row["layer"])
            row["_head"] = int(row["head"])
            row["_pair"] = int(row["pair"])
            row["_alpha_rad"] = float(row["suggested_alpha_rad"])
            rows.append(row)
    return fieldnames, rows


def fake_quant_dequant_q4_0_head(x):
    """
    x: [n_tokens, n_dims], where n_dims is a multiple of 128.
    Returns dequantized float32 tensor with the same shape.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    n_tokens, n_dims = x.shape
    if n_dims % QK4_0_HEAD != 0:
        raise ValueError(f"Expected n_dims multiple of {QK4_0_HEAD}, got {n_dims}")

    n_blocks = n_dims // QK4_0_HEAD
    out = np.empty_like(x, dtype=np.float32)

    for bi in range(n_blocks):
        lo = bi * QK4_0_HEAD
        hi = lo + QK4_0_HEAD
        xb = x[:, lo:hi].astype(np.float32, copy=False)

        arg_absmax = np.abs(xb).argmax(axis=1)
        maxv = xb[np.arange(n_tokens), arg_absmax].astype(np.float32, copy=False)

        d = maxv / -8.0
        inv_d = np.zeros_like(d, dtype=np.float32)
        nz = d != 0
        inv_d[nz] = 1.0 / d[nz]

        scaled = xb * inv_d[:, None]
        q = np.trunc(scaled + 8.5).astype(np.int16)
        q = np.minimum(q, 15)
        q = np.maximum(q, 0)

        out[:, lo:hi] = (q.astype(np.float32) - 8.0) * d[:, None]

    return out


def sse_per_token(x, x_hat):
    diff = x.astype(np.float32, copy=False) - x_hat.astype(np.float32, copy=False)
    return np.sum(diff * diff, axis=1, dtype=np.float64)


def evaluate_candidates_for_head(post_head, rows_for_head):
    """
    post_head: [n_tokens, head_dim] float32
    rows_for_head: list of CSV rows for the same (layer, head)
    """
    base_deq = fake_quant_dequant_q4_0_head(post_head)
    base_sse = sse_per_token(post_head, base_deq)
    base_mean_sse = float(np.mean(base_sse))

    enriched = []
    for row in rows_for_head:
        pair = row["_pair"]
        alpha = row["_alpha_rad"]
        d0 = 2 * pair
        d1 = d0 + 1
        if d1 >= post_head.shape[1]:
            continue

        c = math.cos(alpha)
        s = math.sin(alpha)

        rotated = post_head.copy()
        x0 = post_head[:, d0]
        x1 = post_head[:, d1]
        rotated[:, d0] = c * x0 - s * x1
        rotated[:, d1] = s * x0 + c * x1

        rot_deq = fake_quant_dequant_q4_0_head(rotated)
        rot_sse = sse_per_token(rotated, rot_deq)
        rot_mean_sse = float(np.mean(rot_sse))

        delta = base_mean_sse - rot_mean_sse
        rel_gain_pct = 100.0 * delta / max(base_mean_sse, 1e-12)

        new_row = dict(row)
        new_row["fakeq_err_before"] = f"{base_mean_sse:.10f}"
        new_row["fakeq_err_after"] = f"{rot_mean_sse:.10f}"
        new_row["fakeq_err_delta"] = f"{delta:.10f}"
        new_row["fakeq_err_gain_pct"] = f"{rel_gain_pct:.6f}"
        enriched.append(new_row)

    return enriched


def iter_ropv_post_layers(path, max_tokens):
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

        print(f"ROPV: version={version}, layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")

        for layer in range(n_layers):
            pre_count = struct.unpack("I", f.read(4))[0]
            if pre_count > 0:
                f.seek(pre_count * 4, 1)

            post_count = struct.unpack("I", f.read(4))[0]
            post = None
            loaded_tokens = 0
            if post_count > 0:
                actual_tokens = post_count // stride
                loaded_tokens = min(actual_tokens, max_tokens)
                read_floats = loaded_tokens * stride
                skip_floats = post_count - read_floats
                post = read_chunk_f32(f, read_floats).reshape(loaded_tokens, n_heads, n_dims)
                if skip_floats > 0:
                    f.seek(skip_floats * 4, 1)

            yield layer, post, loaded_tokens, n_heads, n_dims


def main():
    parser = argparse.ArgumentParser(
        description="Filter rotation CSV using token-wise q4_0_head fake-quant error reduction"
    )
    parser.add_argument("input_csv", help="Input rotation CSV")
    parser.add_argument("input_ropv", help="Input ROPV file with post-RoPE values")
    parser.add_argument("output_csv", help="Filtered output CSV")
    parser.add_argument("--analysis-csv", help="Optional CSV with all rows and fake-quant metrics")
    parser.add_argument("--max-tokens", type=int, default=20480, help="Max tokens per layer to evaluate")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Keep rows only if fakeq_err_delta > min-delta")
    parser.add_argument("--min-gain-pct", type=float, default=0.0,
                        help="Keep rows only if fakeq_err_gain_pct > min-gain-pct")
    args = parser.parse_args()

    fieldnames, rows = load_rotation_csv(args.input_csv)
    print(f"Rotation CSV: {args.input_csv}")
    print(f"  rows={len(rows)}")

    rows_by_layer_head = defaultdict(list)
    for row in rows:
        rows_by_layer_head[(row["_layer"], row["_head"])].append(row)

    all_enriched = []

    for layer, post, loaded_tokens, n_heads, n_dims in iter_ropv_post_layers(args.input_ropv, args.max_tokens):
        layer_keys = [key for key in rows_by_layer_head.keys() if key[0] == layer]
        if not layer_keys:
            continue

        if post is None:
            print(f"  Layer {layer}: no post-RoPE data, skipping")
            continue

        print(f"  Layer {layer}: evaluating {len(layer_keys)} heads with {loaded_tokens} tokens")
        for _, head in sorted(layer_keys):
            post_head = post[:, head, :]
            enriched = evaluate_candidates_for_head(post_head, rows_by_layer_head[(layer, head)])
            all_enriched.extend(enriched)

    output_fieldnames = list(fieldnames) + [
        "fakeq_err_before",
        "fakeq_err_after",
        "fakeq_err_delta",
        "fakeq_err_gain_pct",
    ]

    if args.analysis_csv:
        with open(args.analysis_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            for row in all_enriched:
                clean_row = {k: row.get(k, "") for k in output_fieldnames}
                writer.writerow(clean_row)
        print(f"Saved analysis CSV: {args.analysis_csv}")

    kept = []
    for row in all_enriched:
        delta = float(row["fakeq_err_delta"])
        gain_pct = float(row["fakeq_err_gain_pct"])
        if delta > args.min_delta and gain_pct > args.min_gain_pct:
            kept.append(row)

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in kept:
            clean_row = {k: row.get(k, "") for k in output_fieldnames}
            writer.writerow(clean_row)

    print(f"Saved filtered CSV: {args.output_csv}")
    print(f"Kept rows: {len(kept)} / {len(all_enriched)}")

    if all_enriched:
        deltas = np.array([float(r["fakeq_err_delta"]) for r in all_enriched], dtype=np.float64)
        print(
            "Delta stats: "
            f"min={deltas.min():.10f} "
            f"mean={deltas.mean():.10f} "
            f"median={np.median(deltas):.10f} "
            f"max={deltas.max():.10f}"
        )


if __name__ == "__main__":
    main()
