#!/usr/bin/env python3
"""
Build a full per-pair rotation+scale CSV by combining:
  1. baseline full rotation CSV
  2. baseline RPN scales computed from original ROPV
  3. optional override CSV for selected rows (joint gamma/dphi search output)

The output CSV is suitable for fuse_rotation_scale_csv.py and represents:
  - full baseline rotation for every pair
  - full baseline RPN scale for every pair
  - selected override rows replacing angle/scale on top of the baseline
"""

import argparse
import csv
import struct
from typing import Dict, Tuple

import numpy as np

MAGIC_ROPV = 0x524F5056  # "ROPV"


def read_chunk_f32(f, n_floats: int) -> np.ndarray:
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        raise ValueError(f"Expected {n_floats} float32 values, got {got}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def summarize_pair_l2(l2_per_token: np.ndarray, stat_mode: str, percentile: float, tail_lambda: float) -> np.ndarray:
    l2_max = l2_per_token.max(axis=0)
    if stat_mode == "max":
        return l2_max.astype(np.float32, copy=False)

    q = np.percentile(l2_per_token, percentile, axis=0).astype(np.float32, copy=False)
    if stat_mode == "percentile":
        return q
    if stat_mode == "blend":
        return (q + tail_lambda * (l2_max - q)).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported l2 stat mode: {stat_mode}")


def compute_pair_stats_from_ropv(path: str, max_tokens: int, stat_mode: str,
                                 percentile: float, tail_lambda: float):
    pair_stats: Dict[int, np.ndarray] = {}
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
        n_pairs = n_dims // 2

        print(
            f"ROPV: version={version}, layers={n_layers}, heads={n_heads}, dims={n_dims}, "
            f"tokens={n_tokens}, l2_stat={stat_mode}"
        )

        for layer in range(n_layers):
            pre_count = struct.unpack("I", f.read(4))[0]
            if pre_count > 0:
                pre_tokens = pre_count // stride
                pre_loaded = min(pre_tokens, max_tokens)
                read_floats = pre_loaded * stride
                skip_floats = pre_count - read_floats
                pre = read_chunk_f32(f, read_floats).reshape(pre_loaded, n_heads, n_pairs, 2)
                if skip_floats > 0:
                    f.seek(skip_floats * 4, 1)
                l2_per_token = np.sqrt(np.sum(pre * pre, axis=3, dtype=np.float32), dtype=np.float32)
                pair_stats[layer] = summarize_pair_l2(l2_per_token, stat_mode, percentile, tail_lambda)
            else:
                pair_stats[layer] = np.ones((n_heads, n_pairs), dtype=np.float32)

            post_count = struct.unpack("I", f.read(4))[0]
            if post_count > 0:
                f.seek(post_count * 4, 1)

            if (layer + 1) % 8 == 0:
                print(f"  processed {layer + 1}/{n_layers} layers")

    return pair_stats


def load_csv_rows(path: str):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    return fieldnames, rows


def key_of(row):
    return (int(row["layer"]), int(row["head"]), int(row["pair"]))


def main():
    parser = argparse.ArgumentParser(description="Merge full baseline rotation CSV with baseline RPN scales and selected overrides")
    parser.add_argument("baseline_csv", help="Full baseline rotation CSV")
    parser.add_argument("orig_ropv", help="Original ROPV file for baseline RPN scales")
    parser.add_argument("output_csv", help="Merged output CSV for fuse_rotation_scale_csv.py")
    parser.add_argument("--override-csv", help="Optional override CSV from refine_rotation_scale_csv_by_qk_loss.py")
    parser.add_argument("--rpn-alpha", type=float, default=8.0, help="Baseline RPN alpha (default: 8.0)")
    parser.add_argument("--max-tokens", type=int, default=20480, help="Max ROPV tokens to use for base scales")
    parser.add_argument("--l2-stat", choices=["max", "percentile", "blend"], default="max")
    parser.add_argument("--l2-percentile", type=float, default=99.9)
    parser.add_argument("--tail-lambda", type=float, default=0.1)
    args = parser.parse_args()

    base_fieldnames, base_rows = load_csv_rows(args.baseline_csv)
    override_map = {}
    override_fieldnames = []
    if args.override_csv:
        override_fieldnames, override_rows = load_csv_rows(args.override_csv)
        override_map = {key_of(row): row for row in override_rows}

    pair_stats = compute_pair_stats_from_ropv(
        args.orig_ropv,
        max_tokens=args.max_tokens,
        stat_mode=args.l2_stat,
        percentile=args.l2_percentile,
        tail_lambda=args.tail_lambda,
    )

    extra_fields = [
        "orig_suggested_alpha_rad",
        "orig_suggested_alpha_deg",
        "residual_gamma",
        "delta_phi_deg",
        "delta_phi_rad",
        "base_k_scale",
        "base_q_scale",
        "final_k_scale",
        "final_q_scale",
        "override_applied",
        "qk_loss_score",
        "baseline_logit_mse",
        "baseline_sink_logit_mse",
        "baseline_attention_kl",
        "baseline_margin_mse",
        "baseline_top1_match_rate",
        "baseline_attention_l1",
        "best_logit_mse",
        "best_sink_logit_mse",
        "best_attention_kl",
        "best_margin_mse",
        "best_top1_match_rate",
        "best_attention_l1",
    ]
    output_fieldnames = list(base_fieldnames)
    for field in extra_fields:
        if field not in output_fieldnames:
            output_fieldnames.append(field)

    merged_rows = []
    override_count = 0
    for row in base_rows:
        layer = int(row["layer"])
        head = int(row["head"])
        pair = int(row["pair"])

        stat = float(pair_stats[layer][head, pair])
        stat = max(stat, 1e-8)
        base_k_scale = args.rpn_alpha / stat
        base_q_scale = 1.0 / base_k_scale

        merged = dict(row)
        merged["orig_suggested_alpha_rad"] = row["suggested_alpha_rad"]
        merged["orig_suggested_alpha_deg"] = row["suggested_alpha_deg"]
        merged["residual_gamma"] = "1.0000000000"
        merged["delta_phi_deg"] = "0.0000000000"
        merged["delta_phi_rad"] = "0.0000000000"
        merged["base_k_scale"] = f"{base_k_scale:.10f}"
        merged["base_q_scale"] = f"{base_q_scale:.10f}"
        merged["final_k_scale"] = f"{base_k_scale:.10f}"
        merged["final_q_scale"] = f"{base_q_scale:.10f}"
        merged["override_applied"] = "0"
        merged["qk_loss_score"] = ""
        merged["baseline_logit_mse"] = ""
        merged["baseline_sink_logit_mse"] = ""
        merged["baseline_attention_kl"] = ""
        merged["baseline_margin_mse"] = ""
        merged["baseline_top1_match_rate"] = ""
        merged["baseline_attention_l1"] = ""
        merged["best_logit_mse"] = ""
        merged["best_sink_logit_mse"] = ""
        merged["best_attention_kl"] = ""
        merged["best_margin_mse"] = ""
        merged["best_top1_match_rate"] = ""
        merged["best_attention_l1"] = ""

        key = (layer, head, pair)
        override = override_map.get(key)
        if override is not None:
            merged["suggested_alpha_rad"] = override["suggested_alpha_rad"]
            merged["suggested_alpha_deg"] = override["suggested_alpha_deg"]
            for field in extra_fields:
                if field in override and override[field] != "":
                    merged[field] = override[field]
            merged["override_applied"] = "1"
            override_count += 1

        merged_rows.append(merged)

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({k: row.get(k, "") for k in output_fieldnames})

    print(f"Saved merged CSV: {args.output_csv}")
    print(f"Rows written: {len(merged_rows)}")
    print(f"Override rows applied: {override_count}")


if __name__ == "__main__":
    main()
