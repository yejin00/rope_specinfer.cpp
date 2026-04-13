#!/usr/bin/env python3
"""
Refine per-pair rotation alpha using post-RoPE K activations, K-side RPN, and
token-wise q4_0_head fake-quant error.

For each candidate (layer, head, pair) from the input CSV:
  1. Read pre/post-RoPE K activations from a ROPV dump
  2. Compute K-side RPN scales from pre-RoPE pair L2 norms
  3. Apply K-side RPN to the full 128-d post-RoPE head vector
  4. Search alpha over a small grid derived from the original suggested alpha
  5. Keep the alpha that minimizes mean token-wise q4_0_head reconstruction
     error, optionally requiring non-worse p99 token error
  6. Emit a CSV where suggested_alpha_* is replaced by the refined alpha

This is intended as a more direct proxy for the deployed
rotation -> RPN -> q4_0_head pipeline than centroid/diagonal heuristics.
"""

import argparse
import csv
import math
import struct
from collections import defaultdict

import numpy as np


MAGIC_ROPV = 0x524F5056  # "ROPV"
QK4_0_HEAD = 128


def parse_grid_fracs(text):
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("alpha grid must not be empty")
    if 0.0 not in vals:
        vals.append(0.0)
    vals = sorted(set(vals))
    return vals


def load_rotation_csv(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            row["_layer"] = int(row["layer"])
            row["_head"] = int(row["head"])
            row["_pair"] = int(row["pair"])
            row["_alpha_rad"] = float(row["suggested_alpha_rad"])
            rows.append(row)
    return fieldnames, rows


def read_chunk_f32(f, n_floats):
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        raise ValueError(f"Expected {n_floats} float32 values, got {got}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def iter_ropv_pre_post_layers(path, max_tokens):
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

            loaded = min(pre_loaded, post_loaded)
            if pre is not None and pre.shape[0] != loaded:
                pre = pre[:loaded]
            if post is not None and post.shape[0] != loaded:
                post = post[:loaded]

            yield layer, pre, post, loaded, n_heads, n_dims


def compute_k_rpn_scale_from_pre(pre_head, rpn_alpha, eps=1e-6):
    n_tokens, n_dims = pre_head.shape
    if n_dims % 2 != 0:
        raise ValueError(f"Expected even head dimension, got {n_dims}")

    pairs = pre_head.reshape(n_tokens, n_dims // 2, 2)
    l2_per_token = np.sqrt(np.sum(pairs * pairs, axis=2, dtype=np.float32), dtype=np.float32)
    max_l2 = np.maximum(l2_per_token.max(axis=0), eps)
    k_scale_per_pair = rpn_alpha / max_l2
    return np.repeat(k_scale_per_pair, 2).astype(np.float32, copy=False)


def fake_quant_dequant_q4_0_head(x):
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    n_tokens, n_dims = x.shape
    if n_dims % QK4_0_HEAD != 0:
        raise ValueError(f"Expected n_dims multiple of {QK4_0_HEAD}, got {n_dims}")

    out = np.empty_like(x, dtype=np.float32)

    for lo in range(0, n_dims, QK4_0_HEAD):
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


def evaluate_rows_for_head(pre_head, post_head, rows_for_head, rpn_alpha, grid_fracs,
                           require_nonworse_p99):
    if pre_head.shape != post_head.shape:
        raise ValueError(f"pre/post shape mismatch: {pre_head.shape} vs {post_head.shape}")

    rpn_scale = compute_k_rpn_scale_from_pre(pre_head, rpn_alpha)
    post_rpn = post_head.astype(np.float32, copy=False) * rpn_scale[None, :]

    base_deq = fake_quant_dequant_q4_0_head(post_rpn)
    base_sse = sse_per_token(post_rpn, base_deq)
    base_mean = float(np.mean(base_sse))
    base_p99 = float(np.percentile(base_sse, 99.0))

    enriched = []
    for row in rows_for_head:
        pair = row["_pair"]
        a0 = row["_alpha_rad"]
        d0 = 2 * pair
        d1 = d0 + 1
        if d1 >= post_rpn.shape[1]:
            continue

        best_alpha = 0.0
        best_frac = 0.0
        best_mean = base_mean
        best_p99 = base_p99

        for frac in grid_fracs:
            alpha = frac * a0
            if alpha == 0.0:
                cand_mean = base_mean
                cand_p99 = base_p99
            else:
                c = math.cos(alpha)
                s = math.sin(alpha)

                rotated = post_rpn.copy()
                x0 = post_rpn[:, d0]
                x1 = post_rpn[:, d1]
                rotated[:, d0] = c * x0 - s * x1
                rotated[:, d1] = s * x0 + c * x1

                rot_deq = fake_quant_dequant_q4_0_head(rotated)
                rot_sse = sse_per_token(rotated, rot_deq)
                cand_mean = float(np.mean(rot_sse))
                cand_p99 = float(np.percentile(rot_sse, 99.0))

            if require_nonworse_p99 and cand_p99 > base_p99:
                continue

            if cand_mean < best_mean - 1e-12:
                best_alpha = alpha
                best_frac = frac
                best_mean = cand_mean
                best_p99 = cand_p99

        mean_delta = base_mean - best_mean
        mean_gain_pct = 100.0 * mean_delta / max(base_mean, 1e-12)
        p99_delta = base_p99 - best_p99
        p99_gain_pct = 100.0 * p99_delta / max(base_p99, 1e-12)

        new_row = dict(row)
        new_row["orig_suggested_alpha_rad"] = row["suggested_alpha_rad"]
        new_row["orig_suggested_alpha_deg"] = row["suggested_alpha_deg"]
        new_row["suggested_alpha_rad"] = f"{best_alpha:.10f}"
        new_row["suggested_alpha_deg"] = f"{math.degrees(best_alpha):.10f}"
        new_row["best_alpha_frac"] = f"{best_frac:.6f}"
        new_row["rpn_alpha"] = f"{rpn_alpha:.6f}"
        new_row["fakeq_rpn_err_before"] = f"{base_mean:.10f}"
        new_row["fakeq_rpn_err_after"] = f"{best_mean:.10f}"
        new_row["fakeq_rpn_err_delta"] = f"{mean_delta:.10f}"
        new_row["fakeq_rpn_err_gain_pct"] = f"{mean_gain_pct:.6f}"
        new_row["fakeq_rpn_p99_before"] = f"{base_p99:.10f}"
        new_row["fakeq_rpn_p99_after"] = f"{best_p99:.10f}"
        new_row["fakeq_rpn_p99_delta"] = f"{p99_delta:.10f}"
        new_row["fakeq_rpn_p99_gain_pct"] = f"{p99_gain_pct:.6f}"
        enriched.append(new_row)

    return enriched


def main():
    parser = argparse.ArgumentParser(
        description="Refine rotation alpha using RPN-aware q4_0_head fake-quant error"
    )
    parser.add_argument("input_csv", help="Input candidate rotation CSV")
    parser.add_argument("input_ropv", help="Input ROPV file with pre/post K values")
    parser.add_argument("output_csv", help="Filtered output CSV with refined suggested_alpha")
    parser.add_argument("--analysis-csv", help="Optional CSV with all candidate rows and refined metrics")
    parser.add_argument("--max-tokens", type=int, default=20480, help="Max tokens per layer to evaluate")
    parser.add_argument("--rpn-alpha", type=float, required=True,
                        help="K-side RPN target alpha used in the deployed pipeline (e.g. 8.0)")
    parser.add_argument("--alpha-grid-fracs", default="0,0.25,0.5,0.75,1.0",
                        help="Comma-separated fractions applied to original suggested alpha")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Keep rows only if fakeq_rpn_err_delta > min-delta")
    parser.add_argument("--min-gain-pct", type=float, default=0.0,
                        help="Keep rows only if fakeq_rpn_err_gain_pct > min-gain-pct")
    parser.add_argument("--allow-worse-p99", action="store_true",
                        help="Allow candidates whose p99 token error is worse than alpha=0 baseline")
    args = parser.parse_args()

    grid_fracs = parse_grid_fracs(args.alpha_grid_fracs)
    print(f"Alpha grid fractions: {grid_fracs}")

    fieldnames, rows = load_rotation_csv(args.input_csv)
    print(f"Rotation CSV: {args.input_csv}")
    print(f"  rows={len(rows)}")

    rows_by_layer_head = defaultdict(list)
    for row in rows:
        rows_by_layer_head[(row["_layer"], row["_head"])].append(row)

    all_enriched = []

    for layer, pre, post, loaded_tokens, n_heads, n_dims in iter_ropv_pre_post_layers(args.input_ropv, args.max_tokens):
        layer_keys = [key for key in rows_by_layer_head.keys() if key[0] == layer]
        if not layer_keys:
            continue

        if pre is None or post is None:
            print(f"  Layer {layer}: missing pre/post data, skipping")
            continue

        print(f"  Layer {layer}: evaluating {len(layer_keys)} heads with {loaded_tokens} tokens")
        for _, head in sorted(layer_keys):
            pre_head = pre[:, head, :]
            post_head = post[:, head, :]
            enriched = evaluate_rows_for_head(
                pre_head=pre_head,
                post_head=post_head,
                rows_for_head=rows_by_layer_head[(layer, head)],
                rpn_alpha=args.rpn_alpha,
                grid_fracs=grid_fracs,
                require_nonworse_p99=not args.allow_worse_p99,
            )
            all_enriched.extend(enriched)

    output_fieldnames = list(fieldnames) + [
        "orig_suggested_alpha_rad",
        "orig_suggested_alpha_deg",
        "best_alpha_frac",
        "rpn_alpha",
        "fakeq_rpn_err_before",
        "fakeq_rpn_err_after",
        "fakeq_rpn_err_delta",
        "fakeq_rpn_err_gain_pct",
        "fakeq_rpn_p99_before",
        "fakeq_rpn_p99_after",
        "fakeq_rpn_p99_delta",
        "fakeq_rpn_p99_gain_pct",
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
        delta = float(row["fakeq_rpn_err_delta"])
        gain_pct = float(row["fakeq_rpn_err_gain_pct"])
        alpha = float(row["suggested_alpha_rad"])
        if alpha != 0.0 and delta > args.min_delta and gain_pct > args.min_gain_pct:
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
        deltas = np.array([float(r["fakeq_rpn_err_delta"]) for r in all_enriched], dtype=np.float64)
        gains = np.array([float(r["fakeq_rpn_err_gain_pct"]) for r in all_enriched], dtype=np.float64)
        chosen = np.array([float(r["best_alpha_frac"]) for r in all_enriched], dtype=np.float64)
        print(
            "Delta stats: "
            f"min={deltas.min():.10f} "
            f"mean={deltas.mean():.10f} "
            f"median={np.median(deltas):.10f} "
            f"max={deltas.max():.10f}"
        )
        print(
            "Gain pct stats: "
            f"min={gains.min():.6f} "
            f"mean={gains.mean():.6f} "
            f"median={np.median(gains):.6f} "
            f"max={gains.max():.6f}"
        )
        unique_fracs, counts = np.unique(chosen, return_counts=True)
        print("Chosen alpha fractions:")
        for frac, count in zip(unique_fracs, counts):
            print(f"  {frac:.6f}: {count}")


if __name__ == "__main__":
    main()
