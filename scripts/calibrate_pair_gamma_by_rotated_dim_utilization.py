#!/usr/bin/env python3
"""
Calibrate per-pair residual gamma on top of base RPN scales, while keeping
the diagonal rotation angle fixed.

For each late pair row in the input CSV:
  1. Read the original ROPV dump
  2. Compute the base pre-RoPE pair L2 statistic m_p used by RPN
  3. Rotate the post-RoPE pair by the fixed suggested alpha from the CSV
  4. Measure per-channel magnitude after rotation using a configurable stat
  5. Choose gamma so that the larger of the two rotated channels lands on the
     target after applying the pair-common scale:

        final_k_scale = gamma_p * (target / m_p)
        gamma_p       = target / max(base_scaled_dim0, base_scaled_dim1)

This preserves the pair-common-scale constraint: both channels in the pair
share the same gamma and the same base RPN scale.

The output CSV is intended to be consumed by fuse_rpn_alpha.py via --gamma-csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from typing import Dict, List

import numpy as np

MAGIC_ROPV = 0x524F5056  # "ROPV"


def load_rows(path: str) -> tuple[list[str], list[dict]]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            row["_layer"] = int(row["layer"])
            row["_head"] = int(row["head"])
            row["_pair"] = int(row["pair"])
            row["_alpha_rad"] = float(row["suggested_alpha_rad"])
            rows.append(row)
    return fieldnames, rows


def summarize_pair_l2(l2_per_token: np.ndarray, stat_mode: str, percentile: float, tail_lambda: float) -> np.ndarray:
    l2_max = l2_per_token.max(axis=0)
    if stat_mode == "max":
        return l2_max.astype(np.float32, copy=False)

    q = np.percentile(l2_per_token, percentile, axis=0).astype(np.float32, copy=False)
    if stat_mode == "percentile":
        return q
    if stat_mode == "blend":
        return (q + tail_lambda * (l2_max - q)).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported stat mode: {stat_mode}")


def summarize_abs_channels(abs_vals: np.ndarray, stat_mode: str, percentile: float, tail_lambda: float) -> np.ndarray:
    ch_max = abs_vals.max(axis=0)
    if stat_mode == "max":
        return ch_max.astype(np.float32, copy=False)

    q = np.percentile(abs_vals, percentile, axis=0).astype(np.float32, copy=False)
    if stat_mode == "percentile":
        return q
    if stat_mode == "blend":
        return (q + tail_lambda * (ch_max - q)).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported stat mode: {stat_mode}")


def read_chunk_f32(f, n_floats: int) -> np.ndarray:
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        raise ValueError(f"Expected {n_floats} float32 values, got {got}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def rotate_pair(post_pair: np.ndarray, alpha_rad: float) -> np.ndarray:
    c = math.cos(alpha_rad)
    s = math.sin(alpha_rad)
    x = post_pair[:, 0]
    y = post_pair[:, 1]
    out = np.empty_like(post_pair, dtype=np.float32)
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    return out


def build_output_fieldnames(base_fieldnames: list[str]) -> list[str]:
    extra = [
        "target_alpha",
        "base_pair_stat",
        "base_k_scale",
        "base_q_scale",
        "rot_dim_stat0",
        "rot_dim_stat1",
        "base_scaled_dim0",
        "base_scaled_dim1",
        "residual_gamma",
        "final_k_scale",
        "final_q_scale",
        "final_dim0",
        "final_dim1",
        "dominant_channel",
        "dominant_before",
        "target_after",
    ]
    out = list(base_fieldnames)
    for field in extra:
        if field not in out:
            out.append(field)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate per-pair residual gamma using fixed diagonal rotation and rotated post-RoPE dim utilization"
    )
    parser.add_argument("baseline_csv", help="Baseline late-pair rotation CSV (angles fixed)")
    parser.add_argument("ropv_path", help="Original ROPV dump with pre/post values")
    parser.add_argument("output_csv", help="Output CSV with residual_gamma/final_{q,k}_scale")
    parser.add_argument("--analysis-csv", help="Optional CSV with compact analysis columns")
    parser.add_argument("--target-alpha", type=float, default=8.0, help="RPN target alpha (default: 8.0)")
    parser.add_argument("--max-tokens", type=int, default=20480, help="Max ROPV tokens to load per layer")
    parser.add_argument("--pair-stat", choices=["max", "percentile", "blend"], default="max",
                        help="Statistic for pre-RoPE pair L2 used in base RPN scale")
    parser.add_argument("--pair-percentile", type=float, default=99.9,
                        help="Percentile used when --pair-stat is percentile/blend")
    parser.add_argument("--pair-tail-lambda", type=float, default=0.1,
                        help="Blend factor when --pair-stat is blend")
    parser.add_argument("--dim-stat", choices=["max", "percentile", "blend"], default="max",
                        help="Statistic for rotated post-RoPE per-channel magnitudes")
    parser.add_argument("--dim-percentile", type=float, default=99.9,
                        help="Percentile used when --dim-stat is percentile/blend")
    parser.add_argument("--dim-tail-lambda", type=float, default=0.1,
                        help="Blend factor when --dim-stat is blend")
    parser.add_argument("--min-gamma", type=float, default=0.5, help="Lower clamp for residual gamma")
    parser.add_argument("--max-gamma", type=float, default=2.0, help="Upper clamp for residual gamma")
    args = parser.parse_args()

    base_fieldnames, rows = load_rows(args.baseline_csv)
    rows_by_layer: Dict[int, List[dict]] = {}
    for row in rows:
        rows_by_layer.setdefault(row["_layer"], []).append(row)

    output_fieldnames = build_output_fieldnames(base_fieldnames)
    merged_rows: list[dict] = []
    analysis_rows: list[dict] = []
    all_gamma: list[float] = []

    with open(args.ropv_path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]
        n_tokens = struct.unpack("I", f.read(4))[0]
        n_pairs = n_dims // 2
        stride = n_heads * n_dims

        print(
            f"ROPV: version={version}, layers={n_layers}, heads={n_heads}, dims={n_dims}, "
            f"tokens={n_tokens}, target_alpha={args.target_alpha}"
        )
        print(
            f"  pair_stat={args.pair_stat}, dim_stat={args.dim_stat}, "
            f"gamma_clip=[{args.min_gamma}, {args.max_gamma}]"
        )

        for layer in range(n_layers):
            pre_count = struct.unpack("I", f.read(4))[0]
            if pre_count > 0:
                pre_tokens = pre_count // stride
                pre_loaded = min(pre_tokens, args.max_tokens)
                pre_read = pre_loaded * stride
                pre_skip = pre_count - pre_read
                pre = read_chunk_f32(f, pre_read).reshape(pre_loaded, n_heads, n_pairs, 2)
                if pre_skip > 0:
                    f.seek(pre_skip * 4, 1)
            else:
                pre_loaded = 0
                pre = np.zeros((0, n_heads, n_pairs, 2), dtype=np.float32)

            post_count = struct.unpack("I", f.read(4))[0]
            if post_count > 0:
                post_tokens = post_count // stride
                post_loaded = min(post_tokens, args.max_tokens)
                post_read = post_loaded * stride
                post_skip = post_count - post_read
                post = read_chunk_f32(f, post_read).reshape(post_loaded, n_heads, n_pairs, 2)
                if post_skip > 0:
                    f.seek(post_skip * 4, 1)
            else:
                post_loaded = 0
                post = np.zeros((0, n_heads, n_pairs, 2), dtype=np.float32)

            layer_rows = rows_by_layer.get(layer, [])
            if not layer_rows:
                continue

            if pre_loaded == 0 or post_loaded == 0:
                raise ValueError(f"Layer {layer} has no loaded tokens")

            l2_per_token = np.sqrt(np.sum(pre * pre, axis=3, dtype=np.float32), dtype=np.float32)
            pair_stats = summarize_pair_l2(
                l2_per_token,
                stat_mode=args.pair_stat,
                percentile=args.pair_percentile,
                tail_lambda=args.pair_tail_lambda,
            )

            for row in layer_rows:
                head = row["_head"]
                pair = row["_pair"]
                alpha = row["_alpha_rad"]

                pair_stat = float(max(pair_stats[head, pair], 1e-8))
                base_k_scale = args.target_alpha / pair_stat
                base_q_scale = 1.0 / base_k_scale

                rotated = rotate_pair(post[:, head, pair, :], alpha)
                rot_dim_stats = summarize_abs_channels(
                    np.abs(rotated, dtype=np.float32),
                    stat_mode=args.dim_stat,
                    percentile=args.dim_percentile,
                    tail_lambda=args.dim_tail_lambda,
                )
                stat0 = float(rot_dim_stats[0])
                stat1 = float(rot_dim_stats[1])

                base_scaled_dim0 = base_k_scale * stat0
                base_scaled_dim1 = base_k_scale * stat1
                dominant_before = max(base_scaled_dim0, base_scaled_dim1, 1e-8)
                gamma = args.target_alpha / dominant_before
                gamma = float(np.clip(gamma, args.min_gamma, args.max_gamma))

                final_k_scale = base_k_scale * gamma
                final_q_scale = 1.0 / final_k_scale
                final_dim0 = final_k_scale * stat0
                final_dim1 = final_k_scale * stat1
                dominant_channel = 0 if base_scaled_dim0 >= base_scaled_dim1 else 1

                merged = dict(row)
                for k in list(merged.keys()):
                    if k.startswith("_"):
                        del merged[k]
                merged["target_alpha"] = f"{args.target_alpha:.10f}"
                merged["base_pair_stat"] = f"{pair_stat:.10f}"
                merged["base_k_scale"] = f"{base_k_scale:.10f}"
                merged["base_q_scale"] = f"{base_q_scale:.10f}"
                merged["rot_dim_stat0"] = f"{stat0:.10f}"
                merged["rot_dim_stat1"] = f"{stat1:.10f}"
                merged["base_scaled_dim0"] = f"{base_scaled_dim0:.10f}"
                merged["base_scaled_dim1"] = f"{base_scaled_dim1:.10f}"
                merged["residual_gamma"] = f"{gamma:.10f}"
                merged["final_k_scale"] = f"{final_k_scale:.10f}"
                merged["final_q_scale"] = f"{final_q_scale:.10f}"
                merged["final_dim0"] = f"{final_dim0:.10f}"
                merged["final_dim1"] = f"{final_dim1:.10f}"
                merged["dominant_channel"] = str(dominant_channel)
                merged["dominant_before"] = f"{dominant_before:.10f}"
                merged["target_after"] = f"{max(final_dim0, final_dim1):.10f}"
                merged_rows.append(merged)

                analysis_rows.append({
                    "layer": layer,
                    "head": head,
                    "pair": pair,
                    "suggested_alpha_deg": math.degrees(alpha),
                    "base_pair_stat": pair_stat,
                    "base_k_scale": base_k_scale,
                    "rot_dim_stat0": stat0,
                    "rot_dim_stat1": stat1,
                    "base_scaled_dim0": base_scaled_dim0,
                    "base_scaled_dim1": base_scaled_dim1,
                    "residual_gamma": gamma,
                    "final_dim0": final_dim0,
                    "final_dim1": final_dim1,
                    "dominant_channel": dominant_channel,
                })
                all_gamma.append(gamma)

            if (layer + 1) % 8 == 0:
                print(f"  processed {layer + 1}/{n_layers} layers")

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({k: row.get(k, "") for k in output_fieldnames})

    if args.analysis_csv and analysis_rows:
        with open(args.analysis_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(analysis_rows[0].keys()))
            writer.writeheader()
            writer.writerows(analysis_rows)

    g = np.array(all_gamma, dtype=np.float32)
    print(f"Saved gamma CSV: {args.output_csv}")
    if args.analysis_csv and analysis_rows:
        print(f"Saved analysis CSV: {args.analysis_csv}")
    if g.size:
        print(
            f"Residual gamma stats: min={g.min():.6f} mean={g.mean():.6f} "
            f"median={np.median(g):.6f} max={g.max():.6f}"
        )


if __name__ == "__main__":
    main()
