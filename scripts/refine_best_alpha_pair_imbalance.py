#!/usr/bin/env python3
"""
Refine pair-wise best alpha around coarse best_global_alpha_deg by minimizing
pair internal absmax imbalance on post-RoPE values.

Objective:
    gap(alpha) =
        | max_t |x_rot(t)| - max_t |y_rot(t)| |
        / max(max_t |x_rot(t)|, max_t |y_rot(t)|)

The search is done in the canonical 90-degree-equivalent interval and wraps
around +/-45 degrees so boundary coarse optima can still be refined locally.
"""

import argparse
import csv
import math
import os
import struct
from collections import defaultdict

import numpy as np

MAGIC_ROPV = 0x524F5056  # "ROPV"


def parse_priorities(spec):
    return {x.strip() for x in spec.split(",") if x.strip()}


def load_selected_override_rows(path, allowed_priorities):
    rows = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("override_priority") not in allowed_priorities:
                continue
            best_alpha_raw = row.get("best_global_alpha_deg", "")
            if best_alpha_raw == "":
                continue
            rows.append(
                {
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "pair": int(row["pair"]),
                    "dim0": int(row["dim0"]),
                    "dim1": int(row["dim1"]),
                    "override_priority": row["override_priority"],
                    "coarse_best_alpha_deg": float(best_alpha_raw),
                    "coarse_best_gap_csv": float(row.get("best_global_gap", "") or math.nan),
                    "post_residual_gap": float(row.get("post_residual_gap", "") or math.nan),
                    "support_count_234": int(row.get("support_count_234", "") or 0),
                    "criterion1_residual_gap": int(row.get("criterion1_residual_gap", "") or 0),
                    "criterion2_pre_line_like": int(row.get("criterion2_pre_line_like", "") or 0),
                    "criterion3_post_directional": int(row.get("criterion3_post_directional", "") or 0),
                    "criterion4_bucket_consistent": int(row.get("criterion4_bucket_consistent", "") or 0),
                }
            )
    return rows


def canonicalize_deg(alpha_deg):
    wrapped = ((float(alpha_deg) + 45.0) % 90.0) - 45.0
    if math.isclose(wrapped, -45.0, abs_tol=1e-9) and float(alpha_deg) > 0.0:
        return 45.0
    return wrapped


def canonical_delta_deg(dst_deg, src_deg):
    delta = ((float(dst_deg) - float(src_deg) + 45.0) % 90.0) - 45.0
    if math.isclose(delta, -45.0, abs_tol=1e-9) and (float(dst_deg) - float(src_deg)) > 0.0:
        return 45.0
    return delta


def build_refine_grid_deg(center_deg, window_deg, step_deg):
    n_steps = int(round((2.0 * window_deg) / step_deg))
    offsets = np.linspace(-window_deg, window_deg, n_steps + 1, dtype=np.float64)
    values = [canonicalize_deg(center_deg + offset) for offset in offsets]
    values.append(canonicalize_deg(center_deg))

    dedup = {}
    for value in values:
        dedup[round(value, 10)] = float(value)

    grid = np.asarray(sorted(dedup.values()), dtype=np.float32)
    center = canonicalize_deg(center_deg)
    if not np.any(np.isclose(grid, center, atol=1e-6)):
        grid = np.sort(np.append(grid, np.float32(center)))
    return grid


def sweep_absmax_gap_deg(x, y, alpha_deg_grid):
    alpha_rad = np.deg2rad(alpha_deg_grid.astype(np.float32))
    cos_grid = np.cos(alpha_rad, dtype=np.float32)
    sin_grid = np.sin(alpha_rad, dtype=np.float32)

    x = x[None, :]
    y = y[None, :]
    xr = cos_grid[:, None] * x - sin_grid[:, None] * y
    yr = sin_grid[:, None] * x + cos_grid[:, None] * y
    max_x = np.max(np.abs(xr), axis=1)
    max_y = np.max(np.abs(yr), axis=1)
    return np.abs(max_x - max_y) / np.maximum(np.maximum(max_x, max_y), 1e-12)


def refine_rows_from_ropv(ropv_path, selected_rows, window_deg, step_deg):
    by_layer = defaultdict(list)
    for row in selected_rows:
        by_layer[row["layer"]].append(row)

    out = []

    with open(ropv_path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid ROPV magic: {hex(magic)}")

        version, n_layers, n_heads, n_dims, n_tokens = struct.unpack("<5I", f.read(20))

        for layer in range(n_layers):
            pre_count = struct.unpack("<I", f.read(4))[0]
            if pre_count > 0:
                f.seek(pre_count * 4, 1)

            post_count = struct.unpack("<I", f.read(4))[0]
            if post_count > 0:
                post_data = np.frombuffer(f.read(post_count * 4), dtype=np.float32).reshape(-1, n_heads, n_dims)
            else:
                post_data = None

            target_rows = by_layer.get(layer)
            if not target_rows or post_data is None:
                continue

            print(f"Refining layer {layer}: {len(target_rows)} pairs")
            for row in target_rows:
                head = row["head"]
                d0 = row["dim0"]
                d1 = row["dim1"]
                pair_data = post_data[:, head, [d0, d1]].astype(np.float32, copy=False)
                x = pair_data[:, 0]
                y = pair_data[:, 1]

                coarse_best_alpha_deg = row["coarse_best_alpha_deg"]
                search_grid_deg = build_refine_grid_deg(coarse_best_alpha_deg, window_deg, step_deg)
                gaps = sweep_absmax_gap_deg(x, y, search_grid_deg)

                refined_idx = int(np.argmin(gaps))
                refined_best_alpha_deg = float(search_grid_deg[refined_idx])
                refined_best_gap = float(gaps[refined_idx])

                coarse_eval_grid = np.asarray([canonicalize_deg(coarse_best_alpha_deg)], dtype=np.float32)
                coarse_best_gap_recomputed = float(sweep_absmax_gap_deg(x, y, coarse_eval_grid)[0])
                zero_gap = float(sweep_absmax_gap_deg(x, y, np.asarray([0.0], dtype=np.float32))[0])

                out.append(
                    {
                        "layer": row["layer"],
                        "head": row["head"],
                        "pair": row["pair"],
                        "dim0": row["dim0"],
                        "dim1": row["dim1"],
                        "override_priority": row["override_priority"],
                        "search_window_deg": float(window_deg),
                        "search_step_deg": float(step_deg),
                        "search_grid_size": int(search_grid_deg.size),
                        "search_grid_min_deg": float(np.min(search_grid_deg)),
                        "search_grid_max_deg": float(np.max(search_grid_deg)),
                        "coarse_best_alpha_deg": coarse_best_alpha_deg,
                        "coarse_best_alpha_rad": math.radians(coarse_best_alpha_deg),
                        "coarse_best_gap_csv": row["coarse_best_gap_csv"],
                        "coarse_best_gap_recomputed": coarse_best_gap_recomputed,
                        "refined_best_alpha_deg": refined_best_alpha_deg,
                        "refined_best_alpha_rad": math.radians(refined_best_alpha_deg),
                        "refined_best_gap": refined_best_gap,
                        "gap_at_zero_deg": zero_gap,
                        "refine_delta_deg": canonical_delta_deg(refined_best_alpha_deg, coarse_best_alpha_deg),
                        "gap_improvement_vs_coarse": coarse_best_gap_recomputed - refined_best_gap,
                        "gap_improvement_vs_coarse_pct": (
                            100.0 * (coarse_best_gap_recomputed - refined_best_gap) / max(coarse_best_gap_recomputed, 1e-12)
                        ),
                        "gap_improvement_vs_zero": zero_gap - refined_best_gap,
                        "gap_improvement_vs_zero_pct": (
                            100.0 * (zero_gap - refined_best_gap) / max(zero_gap, 1e-12)
                        ),
                        "post_residual_gap": row["post_residual_gap"],
                        "support_count_234": row["support_count_234"],
                        "criterion1_residual_gap": row["criterion1_residual_gap"],
                        "criterion2_pre_line_like": row["criterion2_pre_line_like"],
                        "criterion3_post_directional": row["criterion3_post_directional"],
                        "criterion4_bucket_consistent": row["criterion4_bucket_consistent"],
                    }
                )

    out.sort(
        key=lambda r: (
            {"high": 0, "medium": 1}.get(r["override_priority"], 2),
            -float(r["gap_improvement_vs_zero"]),
            -float(r["gap_improvement_vs_coarse"]),
            r["layer"],
            r["head"],
            r["pair"],
        )
    )
    return out


def save_csv(rows, path):
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Refine per-pair best alpha by minimizing absmax imbalance")
    ap.add_argument("override_csv", help="override_candidates.csv")
    ap.add_argument("ropv_path", help="ROPV full dump path")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--output-name", default="refined_best_alpha.csv")
    ap.add_argument("--priorities", default="high,medium")
    ap.add_argument("--window-deg", type=float, default=5.0)
    ap.add_argument("--step-deg", type=float, default=0.5)
    args = ap.parse_args()

    if args.window_deg <= 0.0:
        raise ValueError("--window-deg must be > 0")
    if args.step_deg <= 0.0:
        raise ValueError("--step-deg must be > 0")

    allowed_priorities = parse_priorities(args.priorities)
    os.makedirs(args.output_dir, exist_ok=True)

    selected_rows = load_selected_override_rows(args.override_csv, allowed_priorities)
    print(f"Selected rows to refine: {len(selected_rows)}")
    if not selected_rows:
        raise ValueError("No selected rows found for refinement")

    refined_rows = refine_rows_from_ropv(
        args.ropv_path,
        selected_rows,
        args.window_deg,
        args.step_deg,
    )
    output_path = os.path.join(args.output_dir, args.output_name)
    save_csv(refined_rows, output_path)

    changed = sum(
        1
        for row in refined_rows
        if not math.isclose(row["coarse_best_alpha_deg"], row["refined_best_alpha_deg"], abs_tol=1e-6)
    )
    mean_gain = float(np.mean([row["gap_improvement_vs_coarse"] for row in refined_rows]))
    print(f"Saved: {output_path}")
    print(f"Pairs refined: {len(refined_rows)}")
    print(f"Pairs with changed alpha: {changed}")
    print(f"Mean gap improvement vs coarse: {mean_gain:.8f}")


if __name__ == "__main__":
    main()
