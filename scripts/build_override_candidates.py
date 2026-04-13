#!/usr/bin/env python3
"""
Build override candidate CSVs by joining:
  1. shape/orientation CSV
  2. APOR residual gap CSV
  3. bucket alpha consistency CSV

The bucket alpha sweep is only run on a provisional shortlist to keep runtime
reasonable:
  residual_gap >= residual_thresh AND (pre_line_like OR post_directional)
"""

import argparse
import csv
import math
import os
import struct
from collections import defaultdict

import numpy as np

MAGIC_APOR = 0x524F5041  # "APOR"
MAGIC_ROPV = 0x524F5056  # "ROPV"


def load_shape_rows(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["layer"] = int(row["layer"])
            row["head"] = int(row["head"])
            row["pair"] = int(row["pair"])
            row["dim0"] = int(row["dim0"])
            row["dim1"] = int(row["dim1"])
            for key in (
                "pre_anisotropy",
                "pre_mean_compactness",
                "pre_balance",
                "post_anisotropy",
                "post_mean_compactness",
                "post_balance",
                "suggested_alpha_deg",
                "suggested_alpha_rad",
            ):
                if key in row and row[key] != "":
                    row[key] = float(row[key])
                else:
                    row[key] = math.nan
            rows.append(row)
    return rows


def load_apor(path):
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != MAGIC_APOR:
            raise ValueError(f"Invalid APOR magic: {hex(magic)}")

        version, n_layers, n_heads, n_dims = struct.unpack("<4I", f.read(16))
        pre = np.zeros((n_layers, n_heads, n_dims), dtype=np.float32)
        post = np.zeros((n_layers, n_heads, n_dims), dtype=np.float32)
        for layer in range(n_layers):
            for head in range(n_heads):
                pre[layer, head] = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                post[layer, head] = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
    return {
        "version": version,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_dims": n_dims,
        "pre": pre,
        "post": post,
    }


def residual_gap(a, b, eps=1e-12):
    return abs(float(a) - float(b)) / max(float(a), float(b), eps)


def build_residual_rows(shape_rows, apor):
    rows = []
    for row in shape_rows:
        layer = row["layer"]
        head = row["head"]
        d0 = row["dim0"]
        d1 = row["dim1"]
        pre0 = float(apor["pre"][layer, head, d0])
        pre1 = float(apor["pre"][layer, head, d1])
        post0 = float(apor["post"][layer, head, d0])
        post1 = float(apor["post"][layer, head, d1])
        rows.append(
            {
                "layer": layer,
                "head": head,
                "pair": row["pair"],
                "dim0": d0,
                "dim1": d1,
                "pre_absmax_dim0": pre0,
                "pre_absmax_dim1": pre1,
                "pre_residual_gap": residual_gap(pre0, pre1),
                "post_absmax_dim0": post0,
                "post_absmax_dim1": post1,
                "post_residual_gap": residual_gap(post0, post1),
                "residual_improvement": residual_gap(pre0, pre1) - residual_gap(post0, post1),
            }
        )
    return rows


def save_csv(rows, path):
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def provisional_keyset(shape_rows, residual_rows, residual_thresh, pre_aniso_thresh, post_aniso_thresh):
    res_map = {(r["layer"], r["head"], r["pair"]): r for r in residual_rows}
    keys = set()
    for row in shape_rows:
        key = (row["layer"], row["head"], row["pair"])
        res = res_map[key]["post_residual_gap"]
        pre_line_like = row["pre_anisotropy"] >= pre_aniso_thresh
        post_directional = row["post_anisotropy"] >= post_aniso_thresh
        if res >= residual_thresh and (pre_line_like or post_directional):
            keys.add(key)
    return keys


def read_bucket_candidates_by_layer(ropv_path, target_keys, bucket_size, alpha_grid_deg):
    if not target_keys:
        return []

    by_layer = defaultdict(list)
    for layer, head, pair in sorted(target_keys):
        by_layer[layer].append((head, pair))

    rows = []
    alpha_grid_rad = np.deg2rad(np.asarray(alpha_grid_deg, dtype=np.float32))
    cos_grid = np.cos(alpha_grid_rad).astype(np.float32)
    sin_grid = np.sin(alpha_grid_rad).astype(np.float32)

    with open(ropv_path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid ROPV magic: {hex(magic)}")

        version, n_layers, n_heads, n_dims, n_tokens = struct.unpack("<5I", f.read(20))
        stride = n_heads * n_dims

        for layer in range(n_layers):
            pre_count = struct.unpack("<I", f.read(4))[0]
            if pre_count > 0:
                f.seek(pre_count * 4, 1)

            post_count = struct.unpack("<I", f.read(4))[0]
            if post_count > 0:
                post_data = np.frombuffer(f.read(post_count * 4), dtype=np.float32).reshape(-1, n_heads, n_dims)
            else:
                post_data = None

            if layer not in by_layer or post_data is None:
                continue

            for head, pair in by_layer[layer]:
                d0 = 2 * pair
                d1 = d0 + 1
                pair_data = post_data[:, head, [d0, d1]].astype(np.float32, copy=False)
                x = pair_data[:, 0]
                y = pair_data[:, 1]

                global_gaps = sweep_absmax_gap(x, y, cos_grid, sin_grid)
                best_idx = int(np.argmin(global_gaps))
                best_alpha_deg = float(alpha_grid_deg[best_idx])
                best_gap = float(global_gaps[best_idx])
                current_gap = float(global_gaps[np.where(np.asarray(alpha_grid_deg) == 0.0)[0][0]])

                bucket_best_alphas = []
                bucket_best_gaps = []
                bucket_centers = []
                for start in range(0, len(x), bucket_size):
                    end = min(start + bucket_size, len(x))
                    xb = x[start:end]
                    yb = y[start:end]
                    if xb.size == 0:
                        continue
                    gaps = sweep_absmax_gap(xb, yb, cos_grid, sin_grid)
                    idx = int(np.argmin(gaps))
                    bucket_best_alphas.append(float(alpha_grid_deg[idx]))
                    bucket_best_gaps.append(float(gaps[idx]))
                    bucket_centers.append((start + end - 1) / 2.0)

                bucket_best_alphas = np.asarray(bucket_best_alphas, dtype=np.float32)
                bucket_best_gaps = np.asarray(bucket_best_gaps, dtype=np.float32)

                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "pair": pair,
                        "dim0": d0,
                        "dim1": d1,
                        "bucket_size": bucket_size,
                        "n_buckets": int(len(bucket_best_alphas)),
                        "current_global_gap": current_gap,
                        "best_global_alpha_deg": best_alpha_deg,
                        "best_global_gap": best_gap,
                        "global_gap_reduction": current_gap - best_gap,
                        "global_gap_reduction_pct": 100.0 * (current_gap - best_gap) / max(current_gap, 1e-12),
                        "bucket_best_alpha_mean_deg": float(np.mean(bucket_best_alphas)),
                        "bucket_best_alpha_std_deg": float(np.std(bucket_best_alphas)),
                        "bucket_best_alpha_min_deg": float(np.min(bucket_best_alphas)),
                        "bucket_best_alpha_max_deg": float(np.max(bucket_best_alphas)),
                        "bucket_best_alpha_range_deg": float(np.max(bucket_best_alphas) - np.min(bucket_best_alphas)),
                        "bucket_best_gap_mean": float(np.mean(bucket_best_gaps)),
                        "bucket_best_gap_median": float(np.median(bucket_best_gaps)),
                        "bucket_consistent_flag": int(
                            (np.std(bucket_best_alphas) <= 10.0)
                            and ((np.max(bucket_best_alphas) - np.min(bucket_best_alphas)) <= 20.0)
                        ),
                    }
                )

    return rows


def sweep_absmax_gap(x, y, cos_grid, sin_grid):
    x = x[None, :]
    y = y[None, :]
    xr = cos_grid[:, None] * x - sin_grid[:, None] * y
    yr = sin_grid[:, None] * x + cos_grid[:, None] * y
    max_x = np.max(np.abs(xr), axis=1)
    max_y = np.max(np.abs(yr), axis=1)
    return np.abs(max_x - max_y) / np.maximum(np.maximum(max_x, max_y), 1e-12)


def build_override_rows(shape_rows, residual_rows, bucket_rows,
                        residual_thresh, pre_aniso_thresh, post_aniso_thresh):
    res_map = {(r["layer"], r["head"], r["pair"]): r for r in residual_rows}
    bucket_map = {(r["layer"], r["head"], r["pair"]): r for r in bucket_rows}
    out = []

    for row in shape_rows:
        key = (row["layer"], row["head"], row["pair"])
        res = res_map[key]
        bucket = bucket_map.get(key, {})

        criterion1 = res["post_residual_gap"] >= residual_thresh
        criterion2 = row["pre_anisotropy"] >= pre_aniso_thresh
        criterion3 = row["post_anisotropy"] >= post_aniso_thresh
        criterion4 = bool(bucket.get("bucket_consistent_flag", 0))

        support_count = int(criterion2) + int(criterion3) + int(criterion4)
        if criterion1 and support_count >= 2:
            priority = "high"
        elif criterion1 and support_count >= 1:
            priority = "medium"
        else:
            priority = "skip"

        out_row = {
            "layer": row["layer"],
            "head": row["head"],
            "pair": row["pair"],
            "dim0": row["dim0"],
            "dim1": row["dim1"],
            "shape_label": row.get("shape_label", ""),
            "orientation_label": row.get("orientation_label", ""),
            "pre_anisotropy": row["pre_anisotropy"],
            "post_anisotropy": row["post_anisotropy"],
            "post_balance": row["post_balance"],
            "heuristic_alpha_deg": row["suggested_alpha_deg"],
            "post_absmax_dim0": res["post_absmax_dim0"],
            "post_absmax_dim1": res["post_absmax_dim1"],
            "post_residual_gap": res["post_residual_gap"],
            "criterion1_residual_gap": int(criterion1),
            "criterion2_pre_line_like": int(criterion2),
            "criterion3_post_directional": int(criterion3),
            "criterion4_bucket_consistent": int(criterion4),
            "support_count_234": support_count,
            "bucket_best_alpha_mean_deg": bucket.get("bucket_best_alpha_mean_deg", ""),
            "bucket_best_alpha_std_deg": bucket.get("bucket_best_alpha_std_deg", ""),
            "bucket_best_alpha_range_deg": bucket.get("bucket_best_alpha_range_deg", ""),
            "best_global_alpha_deg": bucket.get("best_global_alpha_deg", ""),
            "best_global_gap": bucket.get("best_global_gap", ""),
            "global_gap_reduction": bucket.get("global_gap_reduction", ""),
            "global_gap_reduction_pct": bucket.get("global_gap_reduction_pct", ""),
            "override_priority": priority,
        }
        out.append(out_row)

    out.sort(
        key=lambda r: (
            {"high": 0, "medium": 1, "skip": 2}[r["override_priority"]],
            -float(r["post_residual_gap"]),
            -int(r["support_count_234"]),
            r["layer"],
            r["head"],
            r["pair"],
        )
    )
    return out


def parse_alpha_grid(spec):
    return [float(x) for x in spec.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Build override candidate CSVs")
    ap.add_argument("shape_csv", help="Shape/orientation CSV")
    ap.add_argument("ropv_path", help="ROPV full dump for bucket alpha sweep")
    ap.add_argument("apor_path", help="APOR absmax dump for residual gaps")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--residual-thresh", type=float, default=0.25)
    ap.add_argument("--pre-aniso-thresh", type=float, default=0.70)
    ap.add_argument("--post-aniso-thresh", type=float, default=0.35)
    ap.add_argument("--bucket-size", type=int, default=256)
    ap.add_argument(
        "--alpha-grid-deg",
        default="-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45",
        help="Comma-separated delta-alpha grid in degrees",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading shape CSV: {args.shape_csv}")
    shape_rows = load_shape_rows(args.shape_csv)

    print(f"Loading APOR: {args.apor_path}")
    apor = load_apor(args.apor_path)
    residual_rows = build_residual_rows(shape_rows, apor)
    residual_csv = os.path.join(args.output_dir, "apor_residual_rotated_noclip.csv")
    save_csv(residual_rows, residual_csv)
    print(f"Saved: {residual_csv}")

    provisional = provisional_keyset(
        shape_rows,
        residual_rows,
        args.residual_thresh,
        args.pre_aniso_thresh,
        args.post_aniso_thresh,
    )
    print(f"Provisional shortlist for bucket sweep: {len(provisional)} pairs")

    alpha_grid_deg = parse_alpha_grid(args.alpha_grid_deg)
    bucket_rows = read_bucket_candidates_by_layer(
        args.ropv_path,
        provisional,
        args.bucket_size,
        alpha_grid_deg,
    )
    bucket_csv = os.path.join(args.output_dir, "bucket_alpha_consistency_rotated_noclip.csv")
    save_csv(bucket_rows, bucket_csv)
    print(f"Saved: {bucket_csv}")

    override_rows = build_override_rows(
        shape_rows,
        residual_rows,
        bucket_rows,
        args.residual_thresh,
        args.pre_aniso_thresh,
        args.post_aniso_thresh,
    )
    override_csv = os.path.join(args.output_dir, "override_candidates.csv")
    save_csv(override_rows, override_csv)
    print(f"Saved: {override_csv}")

    high = sum(1 for r in override_rows if r["override_priority"] == "high")
    med = sum(1 for r in override_rows if r["override_priority"] == "medium")
    print(f"Override summary: high={high}, medium={med}, skip={len(override_rows) - high - med}")


if __name__ == "__main__":
    main()
