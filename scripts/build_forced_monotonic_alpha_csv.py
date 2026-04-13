#!/usr/bin/env python3
"""
Build a forced monotonic per-pair alpha CSV for offline RPN fusion.

Idea:
- keep fixed rotation from the input rotation CSV
- compute a base-alpha-scaled post-RoPE dominant strength per late pair
- within each (layer, head), rank the 32 late pairs by strength (strongest first)
- assign a monotonic alpha schedule across the ranked pairs
  e.g. alpha schedule 1,2,3,4,5,6,7,8 means:
    strongest quartile -> 1, next -> 2, ..., weakest quartile -> 8
- emit residual_gamma = alpha_p / base_alpha so fuse_rpn_alpha.py can apply it

This is a diagnostic / ablation branch for head-global q4_0_head behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

MAGIC_ROPV = 0x524F5056  # "ROPV"


def parse_csv_grid(text: str) -> np.ndarray:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("Alpha schedule is empty")
    return np.array(vals, dtype=np.float32)


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


def build_output_fieldnames(base_fieldnames: Sequence[str]) -> list[str]:
    extra = [
        "base_alpha",
        "pair_alpha",
        "residual_gamma",
        "base_pair_stat",
        "base_k_scale",
        "base_q_scale",
        "final_k_scale",
        "final_q_scale",
        "dominant_absmax_after_base_alpha",
        "dim0_absmax_after_base_alpha",
        "dim1_absmax_after_base_alpha",
        "strength_rank_in_head",
        "strength_bin_in_head",
        "alpha_schedule",
        "alpha_assignment_mode",
    ]
    out = list(base_fieldnames)
    for field in extra:
        if field not in out:
            out.append(field)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forced monotonic per-pair alpha CSV from ranked late-pair strengths")
    parser.add_argument("rotation_csv", help="Input rotation CSV with suggested_alpha_rad")
    parser.add_argument("ropv_path", help="Input ROPV dump")
    parser.add_argument("output_csv", help="Output CSV with pair_alpha + residual_gamma")
    parser.add_argument("--analysis-csv", help="Optional analysis CSV")
    parser.add_argument("--base-alpha", type=float, default=8.0, help="Base RPN alpha (default: 8.0)")
    parser.add_argument("--alpha-schedule", default="1,2,3,4,5,6,7,8",
                        help="Comma-separated monotonic alpha schedule, assigned from strongest to weakest bins")
    parser.add_argument("--max-tokens", type=int, default=20480, help="Max tokens per layer to load")
    parser.add_argument("--pair-stat", choices=["max", "percentile", "blend"], default="max",
                        help="Statistic for pre-RoPE pair L2 used in base RPN scale")
    parser.add_argument("--pair-percentile", type=float, default=99.9,
                        help="Percentile used when --pair-stat is percentile/blend")
    parser.add_argument("--pair-tail-lambda", type=float, default=0.1,
                        help="Blend factor when --pair-stat is blend")
    args = parser.parse_args()

    alpha_schedule = parse_csv_grid(args.alpha_schedule)
    base_fieldnames, rows = load_rows(args.rotation_csv)
    rows_by_layer_head: Dict[tuple[int, int], List[dict]] = defaultdict(list)
    for row in rows:
        rows_by_layer_head[(row["_layer"], row["_head"])] .append(row)

    output_fieldnames = build_output_fieldnames(base_fieldnames)
    out_rows: list[dict] = []
    chosen_counts = defaultdict(int)

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

        print(f"ROPV: version={version}, layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")
        print(f"  base_alpha={args.base_alpha}, alpha_schedule={alpha_schedule.tolist()}, max_tokens={args.max_tokens}")
        print(f"  total rows={len(rows)}")

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

            layer_heads = sorted({head for lyr, head in rows_by_layer_head if lyr == layer})
            if not layer_heads:
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

            print(f"  Layer {layer}: evaluating heads {layer_heads} with {post_loaded} tokens")

            for head in layer_heads:
                rows_for_head = sorted(rows_by_layer_head[(layer, head)], key=lambda r: r["_pair"])
                if not rows_for_head:
                    continue

                post_pairs = post[:, head, :, :].astype(np.float32, copy=True)
                for row in rows_for_head:
                    pair = row["_pair"]
                    post_pairs[:, pair, :] = rotate_pair(post_pairs[:, pair, :], row["_alpha_rad"])

                base_pair_stats = np.maximum(pair_stats[head], 1e-8).astype(np.float32, copy=False)
                base_pair_scales = (args.base_alpha / base_pair_stats).astype(np.float32, copy=False)

                strength_records = []
                for row in rows_for_head:
                    pair = row["_pair"]
                    scaled_pair = post_pairs[:, pair, :] * base_pair_scales[pair]
                    dim0_absmax = float(np.max(np.abs(scaled_pair[:, 0])))
                    dim1_absmax = float(np.max(np.abs(scaled_pair[:, 1])))
                    dominant = max(dim0_absmax, dim1_absmax)
                    strength_records.append({
                        "row": row,
                        "pair": pair,
                        "dim0_absmax": dim0_absmax,
                        "dim1_absmax": dim1_absmax,
                        "dominant": dominant,
                    })

                strength_records.sort(key=lambda rec: (-rec["dominant"], rec["pair"]))
                groups = np.array_split(np.arange(len(strength_records)), len(alpha_schedule))
                assigned = {}
                for bin_idx, (alpha_val, grp) in enumerate(zip(alpha_schedule, groups), start=1):
                    for rank_idx in grp.tolist():
                        assigned[rank_idx] = (float(alpha_val), bin_idx)

                for rank0, rec in enumerate(strength_records):
                    alpha_val, bin_idx = assigned[rank0]
                    pair = rec["pair"]
                    row = rec["row"]
                    pair_stat = float(base_pair_stats[pair])
                    base_k_scale = float(base_pair_scales[pair])
                    base_q_scale = float(1.0 / base_k_scale)
                    residual_gamma = float(alpha_val / args.base_alpha)
                    final_k_scale = float(alpha_val / pair_stat)
                    final_q_scale = float(1.0 / final_k_scale)

                    out = dict(row)
                    for k in list(out.keys()):
                        if k.startswith("_"):
                            del out[k]
                    out["base_alpha"] = f"{args.base_alpha:.10f}"
                    out["pair_alpha"] = f"{alpha_val:.10f}"
                    out["residual_gamma"] = f"{residual_gamma:.10f}"
                    out["base_pair_stat"] = f"{pair_stat:.10f}"
                    out["base_k_scale"] = f"{base_k_scale:.10f}"
                    out["base_q_scale"] = f"{base_q_scale:.10f}"
                    out["final_k_scale"] = f"{final_k_scale:.10f}"
                    out["final_q_scale"] = f"{final_q_scale:.10f}"
                    out["dominant_absmax_after_base_alpha"] = f"{rec['dominant']:.10f}"
                    out["dim0_absmax_after_base_alpha"] = f"{rec['dim0_absmax']:.10f}"
                    out["dim1_absmax_after_base_alpha"] = f"{rec['dim1_absmax']:.10f}"
                    out["strength_rank_in_head"] = str(rank0 + 1)
                    out["strength_bin_in_head"] = str(bin_idx)
                    out["alpha_schedule"] = ",".join(f"{float(a):.6f}" for a in alpha_schedule)
                    out["alpha_assignment_mode"] = "forced_monotonic_rank_bins"
                    out_rows.append(out)
                    chosen_counts[alpha_val] += 1

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in output_fieldnames})
    print(f"Saved forced monotonic alpha CSV: {args.output_csv}")

    if args.analysis_csv:
        with open(args.analysis_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            for row in out_rows:
                writer.writerow({k: row.get(k, "") for k in output_fieldnames})
        print(f"Saved analysis CSV: {args.analysis_csv}")

    print("Chosen alpha counts:", dict(sorted(chosen_counts.items())))


if __name__ == "__main__":
    main()
