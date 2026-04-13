#!/usr/bin/env python3
"""
Build a combined rotation CSV for a sink-aware early-band experiment.

The output CSV is intended to be directly consumable by fuse_rotation_csv.py.

Behavior:
  - For pairs in [pair_start, pair_end], recompute alpha from the first
    `sink_tokens` post-RoPE tokens only, aligning the sink centroid to the
    nearest diagonal.
  - For all other pairs, keep rows from the base CSV unchanged.

This is useful for quick "early-dim sink-guided rotation" probes where
late pairs keep an existing rotation policy while early pairs use a
different offline heuristic.
"""

import argparse
import csv
import math
import struct
from collections import defaultdict

import numpy as np


MAGIC_ROPV = 0x524F5056  # "ROPV"

DEFAULT_FIELDS = [
    "layer",
    "head",
    "pair",
    "dim0",
    "dim1",
    "pre_spread",
    "pre_anisotropy",
    "pre_mean_compactness",
    "pre_balance",
    "pre_mu_norm",
    "pre_mu_angle",
    "pre_pca_angle",
    "pre_lam1",
    "pre_lam2",
    "post_spread",
    "post_anisotropy",
    "post_mean_compactness",
    "post_balance",
    "post_mu_norm",
    "post_mu_angle",
    "post_pca_angle",
    "post_lam1",
    "post_lam2",
    "mu_drift",
    "pca_drift",
    "shape_label",
    "shape_source",
    "orientation_source",
    "orientation_label",
    "base_angle_rad",
    "base_angle_deg",
    "diag_distance_rad",
    "diag_distance_deg",
    "axis_distance_rad",
    "axis_distance_deg",
    "target_diag_angle_rad",
    "target_diag_angle_deg",
    "suggested_alpha_rad",
    "suggested_alpha_deg",
]


def wrap_angle(x):
    return float(np.arctan2(np.sin(x), np.cos(x)))


def axis_angle_diff(a, b):
    return 0.5 * np.abs(np.arctan2(np.sin(2 * (a - b)), np.cos(2 * (a - b))))


def nearest_diag_angle(phi):
    diags = np.array([np.pi / 4, -np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4], dtype=np.float64)
    diffs = np.abs(np.arctan2(np.sin(diags - phi), np.cos(diags - phi)))
    return float(diags[np.argmin(diffs)])


def angle_dist_to_diagonal(phi):
    diags = np.array([np.pi / 4, -np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4], dtype=np.float64)
    diffs = np.abs(np.arctan2(np.sin(diags - phi), np.cos(diags - phi)))
    return float(np.min(diffs))


def angle_dist_to_axis(phi):
    axes = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float64)
    diffs = np.abs(np.arctan2(np.sin(axes - phi), np.cos(axes - phi)))
    return float(np.min(diffs))


def compute_pair_shape_stats(x, y, eps=1e-12):
    z = np.stack([x, y], axis=1).astype(np.float64)
    mu = z.mean(axis=0)
    centered = z - mu
    cov = (centered.T @ centered) / max(len(z), 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    lam1 = float(max(eigvals[0], 0.0))
    lam2 = float(max(eigvals[1], 0.0))
    v1 = eigvecs[:, 0]

    spread = lam1 + lam2
    anisotropy = (lam1 - lam2) / (spread + eps)
    mean_norm2 = float(mu @ mu)
    mean_compactness = mean_norm2 / (mean_norm2 + spread + eps)

    absx = np.abs(x)
    absy = np.abs(y)
    balance = 1.0 - np.mean(np.abs(absx - absy) / (absx + absy + eps))

    mu_norm = np.sqrt(mean_norm2)
    mu_angle = float(np.arctan2(mu[1], mu[0])) if mu_norm > 1e-10 else np.nan
    pca_angle = float(np.arctan2(v1[1], v1[0]))

    return {
        "mu_norm": float(mu_norm),
        "mu_angle": mu_angle,
        "pca_angle": pca_angle,
        "spread": float(spread),
        "anisotropy": float(anisotropy),
        "mean_compactness": float(mean_compactness),
        "balance": float(balance),
        "lam1": lam1,
        "lam2": lam2,
    }


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

        print(
            f"ROPV: version={version}, layers={n_layers}, "
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

            loaded = min(pre_loaded, post_loaded)
            if pre is not None and pre.shape[0] != loaded:
                pre = pre[:loaded]
            if post is not None and post.shape[0] != loaded:
                post = post[:loaded]

            yield layer, pre, post, loaded, n_heads, n_dims


def load_base_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fieldnames:
        fieldnames = list(DEFAULT_FIELDS)
    return fieldnames, rows


def classify_shape(rows_for_head, shape_base):
    spreads = np.array([r[f"{shape_base}_spread"] for r in rows_for_head], dtype=np.float64)
    anisos = np.array([r[f"{shape_base}_anisotropy"] for r in rows_for_head], dtype=np.float64)
    compacts = np.array([r[f"{shape_base}_mean_compactness"] for r in rows_for_head], dtype=np.float64)

    tau_spread_low = float(np.percentile(spreads, 35))
    tau_aniso_high = float(np.percentile(anisos, 70))
    tau_compact_high = float(np.percentile(compacts, 70))

    for r in rows_for_head:
        spread = r[f"{shape_base}_spread"]
        aniso = r[f"{shape_base}_anisotropy"]
        compact = r[f"{shape_base}_mean_compactness"]

        if aniso >= tau_aniso_high:
            r["shape_label"] = "elongated"
        elif spread <= tau_spread_low and compact >= tau_compact_high:
            r["shape_label"] = "compact_cluster"
        else:
            r["shape_label"] = "isotropic_scatter"


def build_sink_row(
    layer_idx,
    head_idx,
    pair_idx,
    pre_head,
    post_head,
    sink_tokens,
    diag_thresh_deg,
    axis_thresh_deg,
    alpha_zero_thresh_deg,
):
    d0 = 2 * pair_idx
    d1 = d0 + 1

    x_pre = pre_head[:sink_tokens, d0]
    y_pre = pre_head[:sink_tokens, d1]
    x_post = post_head[:sink_tokens, d0]
    y_post = post_head[:sink_tokens, d1]

    pre_stats = compute_pair_shape_stats(x_pre, y_pre)
    post_stats = compute_pair_shape_stats(x_post, y_post)

    base_angle = post_stats["mu_angle"]
    if np.isfinite(base_angle):
        diag_dist = angle_dist_to_diagonal(base_angle)
        axis_dist = angle_dist_to_axis(base_angle)
        target_diag = nearest_diag_angle(base_angle)
        alpha = wrap_angle(target_diag - base_angle)
        if abs(np.degrees(alpha)) < alpha_zero_thresh_deg:
            alpha = 0.0
    else:
        diag_dist = np.nan
        axis_dist = np.nan
        target_diag = np.nan
        alpha = np.nan

    diag_thresh = math.radians(diag_thresh_deg)
    axis_thresh = math.radians(axis_thresh_deg)
    if np.isfinite(diag_dist) and diag_dist <= diag_thresh:
        orientation_label = "already_diagonal"
    elif np.isfinite(axis_dist) and axis_dist <= axis_thresh:
        orientation_label = "axis_like"
    else:
        orientation_label = "intermediate"

    if np.isfinite(pre_stats["mu_angle"]) and np.isfinite(post_stats["mu_angle"]):
        mu_drift = float(abs(wrap_angle(post_stats["mu_angle"] - pre_stats["mu_angle"])))
    else:
        mu_drift = np.nan

    row = {
        "layer": layer_idx,
        "head": head_idx,
        "pair": pair_idx,
        "dim0": d0,
        "dim1": d1,
        "pre_spread": pre_stats["spread"],
        "pre_anisotropy": pre_stats["anisotropy"],
        "pre_mean_compactness": pre_stats["mean_compactness"],
        "pre_balance": pre_stats["balance"],
        "pre_mu_norm": pre_stats["mu_norm"],
        "pre_mu_angle": pre_stats["mu_angle"],
        "pre_pca_angle": pre_stats["pca_angle"],
        "pre_lam1": pre_stats["lam1"],
        "pre_lam2": pre_stats["lam2"],
        "post_spread": post_stats["spread"],
        "post_anisotropy": post_stats["anisotropy"],
        "post_mean_compactness": post_stats["mean_compactness"],
        "post_balance": post_stats["balance"],
        "post_mu_norm": post_stats["mu_norm"],
        "post_mu_angle": post_stats["mu_angle"],
        "post_pca_angle": post_stats["pca_angle"],
        "post_lam1": post_stats["lam1"],
        "post_lam2": post_stats["lam2"],
        "mu_drift": mu_drift,
        "pca_drift": float(axis_angle_diff(post_stats["pca_angle"], pre_stats["pca_angle"])),
        "shape_label": "",
        "shape_source": f"post_sink{sink_tokens}",
        "orientation_source": f"post_sink{sink_tokens}_mean",
        "orientation_label": orientation_label,
        "base_angle_rad": float(base_angle) if np.isfinite(base_angle) else np.nan,
        "base_angle_deg": float(np.degrees(base_angle)) if np.isfinite(base_angle) else np.nan,
        "diag_distance_rad": float(diag_dist) if np.isfinite(diag_dist) else np.nan,
        "diag_distance_deg": float(np.degrees(diag_dist)) if np.isfinite(diag_dist) else np.nan,
        "axis_distance_rad": float(axis_dist) if np.isfinite(axis_dist) else np.nan,
        "axis_distance_deg": float(np.degrees(axis_dist)) if np.isfinite(axis_dist) else np.nan,
        "target_diag_angle_rad": float(target_diag) if np.isfinite(target_diag) else np.nan,
        "target_diag_angle_deg": float(np.degrees(target_diag)) if np.isfinite(target_diag) else np.nan,
        "suggested_alpha_rad": float(alpha) if np.isfinite(alpha) else np.nan,
        "suggested_alpha_deg": float(np.degrees(alpha)) if np.isfinite(alpha) else np.nan,
    }
    return row


def merge_fieldnames(base_fields):
    fields = list(base_fields)
    for name in DEFAULT_FIELDS:
        if name not in fields:
            fields.append(name)
    return fields


def main():
    parser = argparse.ArgumentParser(
        description="Build sink-aware early-band rotation CSV merged with an existing late CSV."
    )
    parser.add_argument("base_csv", help="Existing rotation CSV to keep for non-overridden pairs")
    parser.add_argument("input_ropv", help="ROPV dump used to derive sink-aware early alpha")
    parser.add_argument("output_csv", help="Output merged CSV")
    parser.add_argument("--pair-start", type=int, default=0, help="First pair to override (inclusive)")
    parser.add_argument("--pair-end", type=int, default=31, help="Last pair to override (inclusive)")
    parser.add_argument("--sink-tokens", type=int, default=4, help="Number of initial tokens to use")
    parser.add_argument("--max-tokens", type=int, default=4, help="How many tokens to read from ROPV")
    parser.add_argument("--diag-thresh-deg", type=float, default=10.0, help="already_diagonal threshold")
    parser.add_argument("--axis-thresh-deg", type=float, default=15.0, help="axis_like threshold")
    parser.add_argument("--alpha-zero-thresh-deg", type=float, default=5.0, help="Zero-out tiny alpha")
    args = parser.parse_args()

    if args.max_tokens < args.sink_tokens:
        raise ValueError("--max-tokens must be >= --sink-tokens")
    if args.pair_start < 0 or args.pair_end < args.pair_start:
        raise ValueError("invalid pair range")

    base_fields, base_rows = load_base_csv(args.base_csv)
    fieldnames = merge_fieldnames(base_fields)

    row_map = {}
    for row in base_rows:
        key = (int(row["layer"]), int(row["head"]), int(row["pair"]))
        row_map[key] = dict(row)

    generated = defaultdict(list)
    for layer, pre, post, loaded, n_heads, n_dims in iter_ropv_pre_post_layers(args.input_ropv, args.max_tokens):
        if pre is None or post is None or loaded < args.sink_tokens:
            continue
        n_pairs = n_dims // 2
        pair_hi = min(args.pair_end, n_pairs - 1)
        for head in range(n_heads):
            pre_head = pre[:, head, :]
            post_head = post[:, head, :]
            for pair in range(args.pair_start, pair_hi + 1):
                row = build_sink_row(
                    layer,
                    head,
                    pair,
                    pre_head,
                    post_head,
                    sink_tokens=args.sink_tokens,
                    diag_thresh_deg=args.diag_thresh_deg,
                    axis_thresh_deg=args.axis_thresh_deg,
                    alpha_zero_thresh_deg=args.alpha_zero_thresh_deg,
                )
                generated[(layer, head)].append(row)

    override_count = 0
    zero_alpha_count = 0
    for key, group in generated.items():
        classify_shape(group, shape_base="post")
        for row in group:
            if float(row["suggested_alpha_rad"]) == 0.0:
                zero_alpha_count += 1
            row_map[(row["layer"], row["head"], row["pair"])] = row
            override_count += 1

    merged_rows = [row_map[k] for k in sorted(row_map.keys())]

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            out = {name: row.get(name, "") for name in fieldnames}
            writer.writerow(out)

    print(f"Wrote merged CSV: {args.output_csv}")
    print(f"  Base rows kept/merged: {len(base_rows)}")
    print(f"  Early rows overridden: {override_count}")
    print(f"  Zeroed alpha rows: {zero_alpha_count}")
    print(f"  Output rows: {len(merged_rows)}")


if __name__ == "__main__":
    main()
