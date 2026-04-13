#!/usr/bin/env python3
"""
Compare post-RoPE APOR absmax values for selected pairs before/after clipping.

Selection is taken from a clip-candidate CSV, typically produced from the
original ROPV using analyze_rope_clip_candidates.py.
"""

import argparse
import csv
import math
import struct

import numpy as np


MAGIC_APOR = 0x524F5041  # "APOR"


def load_apor(path):
    with open(path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_APOR:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected APOR")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]

        pre_absmax = np.zeros((n_layers, n_heads, n_dims), dtype=np.float32)
        post_absmax = np.zeros((n_layers, n_heads, n_dims), dtype=np.float32)

        for layer in range(n_layers):
            for head in range(n_heads):
                pre_absmax[layer, head] = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                post_absmax[layer, head] = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)

    return {
        "version": version,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_dims": n_dims,
        "pre_absmax": pre_absmax,
        "post_absmax": post_absmax,
    }


def load_selected_pairs(csv_path, ratio_thresh):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratio = float(row["ratio_max_q99_9"])
            if ratio < ratio_thresh:
                continue
            rows.append(
                {
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "pair": int(row["pair"]),
                    "dim0": int(row["dim0"]),
                    "dim1": int(row["dim1"]),
                    "ratio_max_q99_9": ratio,
                    "clip_tau": float(row.get("clip_tau", "nan")),
                    "tail_label": row.get("tail_label", ""),
                }
            )
    rows.sort(key=lambda row: (-row["ratio_max_q99_9"], row["layer"], row["head"], row["pair"]))
    return rows


def pct_reduction(orig, new, eps=1e-12):
    denom = max(float(orig), eps)
    return 100.0 * (float(orig) - float(new)) / denom


def build_comparison_rows(selected_rows, orig_apor, clipped_apor):
    rows = []
    for sel in selected_rows:
        layer = sel["layer"]
        head = sel["head"]
        d0 = sel["dim0"]
        d1 = sel["dim1"]

        o0 = float(orig_apor["post_absmax"][layer, head, d0])
        o1 = float(orig_apor["post_absmax"][layer, head, d1])
        c0 = float(clipped_apor["post_absmax"][layer, head, d0])
        c1 = float(clipped_apor["post_absmax"][layer, head, d1])

        # This is only a proxy because channel-wise maxima may come from different tokens.
        pair_proxy_orig = math.sqrt(o0 * o0 + o1 * o1)
        pair_proxy_clip = math.sqrt(c0 * c0 + c1 * c1)

        rows.append(
            {
                "layer": layer,
                "head": head,
                "pair": sel["pair"],
                "dim0": d0,
                "dim1": d1,
                "ratio_max_q99_9": sel["ratio_max_q99_9"],
                "clip_tau": sel["clip_tau"],
                "tail_label": sel["tail_label"],
                "orig_post_absmax_dim0": o0,
                "clip_post_absmax_dim0": c0,
                "dim0_reduction_pct": pct_reduction(o0, c0),
                "orig_post_absmax_dim1": o1,
                "clip_post_absmax_dim1": c1,
                "dim1_reduction_pct": pct_reduction(o1, c1),
                "orig_post_pair_proxy": pair_proxy_orig,
                "clip_post_pair_proxy": pair_proxy_clip,
                "pair_proxy_reduction_pct": pct_reduction(pair_proxy_orig, pair_proxy_clip),
                "max_dim_reduction_pct": max(pct_reduction(o0, c0), pct_reduction(o1, c1)),
            }
        )

    rows.sort(key=lambda row: (-row["pair_proxy_reduction_pct"], -row["max_dim_reduction_pct"], row["layer"], row["head"], row["pair"]))
    return rows


def save_csv(rows, path):
    if not rows:
        print("No rows to save.")
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {path}")


def print_summary(rows, topk):
    print(f"\nCompared selected pairs: {len(rows)}")
    if not rows:
        return

    pair_red = np.array([row["pair_proxy_reduction_pct"] for row in rows], dtype=np.float64)
    dim_red = np.array([row["max_dim_reduction_pct"] for row in rows], dtype=np.float64)
    print(f"  mean pair-proxy reduction: {np.mean(pair_red):.2f}%")
    print(f"  median pair-proxy reduction: {np.median(pair_red):.2f}%")
    print(f"  mean max-dim reduction: {np.mean(dim_red):.2f}%")
    print(f"  median max-dim reduction: {np.median(dim_red):.2f}%")

    print(f"\nTop {min(topk, len(rows))} pairs by pair-proxy reduction:")
    for row in rows[:topk]:
        print(
            f"  L{row['layer']:02d} H{row['head']:02d} P{row['pair']:02d} | "
            f"ratio={row['ratio_max_q99_9']:.3f} | "
            f"d0={row['orig_post_absmax_dim0']:.4f}->{row['clip_post_absmax_dim0']:.4f} "
            f"({row['dim0_reduction_pct']:.2f}%) | "
            f"d1={row['orig_post_absmax_dim1']:.4f}->{row['clip_post_absmax_dim1']:.4f} "
            f"({row['dim1_reduction_pct']:.2f}%) | "
            f"pair_proxy={row['orig_post_pair_proxy']:.4f}->{row['clip_post_pair_proxy']:.4f} "
            f"({row['pair_proxy_reduction_pct']:.2f}%)"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare original vs clipped post absmax for selected pairs")
    parser.add_argument("selection_csv", help="Original clip_candidates.csv used to define selected pairs")
    parser.add_argument("original_apor", help="Original APOR absmax file")
    parser.add_argument("clipped_apor", help="Clipped APOR absmax file")
    parser.add_argument("-o", "--output-csv", required=True, help="Output comparison CSV")
    parser.add_argument("--ratio-thresh", type=float, default=1.5, help="Select pairs with ratio_max_q99_9 >= this threshold")
    parser.add_argument("--topk", type=int, default=20, help="Rows to print in summary")
    args = parser.parse_args()

    selected_rows = load_selected_pairs(args.selection_csv, args.ratio_thresh)
    print(f"Loaded selected pairs from {args.selection_csv}: {len(selected_rows)}")

    orig_apor = load_apor(args.original_apor)
    clipped_apor = load_apor(args.clipped_apor)

    rows = build_comparison_rows(selected_rows, orig_apor, clipped_apor)
    save_csv(rows, args.output_csv)
    print_summary(rows, args.topk)


if __name__ == "__main__":
    main()
