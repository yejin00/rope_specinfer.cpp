#!/usr/bin/env python3
"""
Build apply-ready rotation CSVs from:
  1. baseline full rotation CSV
  2. override_candidates.csv

Outputs:
  - full CSV with suggested_alpha_* overridden for selected rows
  - selected-only CSV for rows that will actually be rotated

If --refined-best-csv is provided, refined_best_alpha_deg is preferred over the
coarse best_global_alpha_deg.
"""

import argparse
import csv
import math
import os


def load_rows(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames)


def key_of(row):
    return (int(row["layer"]), int(row["head"]), int(row["pair"]))


def parse_priorities(spec):
    return {x.strip() for x in spec.split(",") if x.strip()}


def choose_alpha_deg(base_row, override_row, refined_row, allowed_priorities):
    priority = override_row["override_priority"]
    if priority not in allowed_priorities:
        return float(base_row["suggested_alpha_deg"]), "baseline"

    if refined_row is not None:
        refined_deg_raw = refined_row.get("refined_best_alpha_deg", "")
        if refined_deg_raw != "":
            return float(refined_deg_raw), "refined_best"

    best_deg_raw = override_row.get("best_global_alpha_deg", "")
    if best_deg_raw != "":
        return float(best_deg_raw), "best_global"

    return float(base_row["suggested_alpha_deg"]), "heuristic_fallback"


def main():
    ap = argparse.ArgumentParser(description="Build apply-ready rotation CSVs from override candidates")
    ap.add_argument("baseline_csv", help="Baseline full rotation CSV")
    ap.add_argument("override_csv", help="override_candidates.csv")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--refined-best-csv", default="", help="Optional refined_best_alpha.csv")
    ap.add_argument("--priorities", default="high,medium", help="Comma-separated override priorities to apply")
    ap.add_argument("--full-name", default="rotation_apply_full_high_medium.csv")
    ap.add_argument("--selected-name", default="rotation_apply_selected_high_medium.csv")
    args = ap.parse_args()

    allowed_priorities = parse_priorities(args.priorities)
    os.makedirs(args.output_dir, exist_ok=True)

    base_rows, base_fields = load_rows(args.baseline_csv)
    override_rows, _ = load_rows(args.override_csv)
    override_map = {key_of(row): row for row in override_rows}
    refined_map = {}
    if args.refined_best_csv:
        refined_rows, _ = load_rows(args.refined_best_csv)
        refined_map = {key_of(row): row for row in refined_rows}

    extra_fields = [
        "orig_suggested_alpha_rad",
        "orig_suggested_alpha_deg",
        "override_applied",
        "override_priority",
        "alpha_source",
        "best_global_alpha_deg",
        "refined_best_alpha_deg",
        "refined_best_gap",
        "refine_delta_deg",
        "gap_improvement_vs_coarse",
        "post_residual_gap",
        "support_count_234",
        "criterion1_residual_gap",
        "criterion2_pre_line_like",
        "criterion3_post_directional",
        "criterion4_bucket_consistent",
        "bucket_best_alpha_std_deg",
        "bucket_best_alpha_range_deg",
        "global_gap_reduction_pct",
    ]

    output_fields = list(base_fields)
    for field in extra_fields:
        if field not in output_fields:
            output_fields.append(field)

    full_rows = []
    selected_rows = []
    applied = 0

    for base_row in base_rows:
        key = key_of(base_row)
        ov = override_map.get(key)
        refined = refined_map.get(key)

        out = dict(base_row)
        out["orig_suggested_alpha_rad"] = base_row["suggested_alpha_rad"]
        out["orig_suggested_alpha_deg"] = base_row["suggested_alpha_deg"]
        out["override_applied"] = "0"
        out["override_priority"] = ov["override_priority"] if ov is not None else "skip"
        out["alpha_source"] = "baseline"

        if ov is not None:
            chosen_deg, alpha_source = choose_alpha_deg(base_row, ov, refined, allowed_priorities)
            out["alpha_source"] = alpha_source
            out["best_global_alpha_deg"] = ov.get("best_global_alpha_deg", "")
            out["refined_best_alpha_deg"] = refined.get("refined_best_alpha_deg", "") if refined is not None else ""
            out["refined_best_gap"] = refined.get("refined_best_gap", "") if refined is not None else ""
            out["refine_delta_deg"] = refined.get("refine_delta_deg", "") if refined is not None else ""
            out["gap_improvement_vs_coarse"] = (
                refined.get("gap_improvement_vs_coarse", "") if refined is not None else ""
            )
            out["post_residual_gap"] = ov.get("post_residual_gap", "")
            out["support_count_234"] = ov.get("support_count_234", "")
            out["criterion1_residual_gap"] = ov.get("criterion1_residual_gap", "")
            out["criterion2_pre_line_like"] = ov.get("criterion2_pre_line_like", "")
            out["criterion3_post_directional"] = ov.get("criterion3_post_directional", "")
            out["criterion4_bucket_consistent"] = ov.get("criterion4_bucket_consistent", "")
            out["bucket_best_alpha_std_deg"] = ov.get("bucket_best_alpha_std_deg", "")
            out["bucket_best_alpha_range_deg"] = ov.get("bucket_best_alpha_range_deg", "")
            out["global_gap_reduction_pct"] = ov.get("global_gap_reduction_pct", "")

            if ov["override_priority"] in allowed_priorities:
                out["suggested_alpha_deg"] = f"{chosen_deg:.10f}"
                out["suggested_alpha_rad"] = f"{math.radians(chosen_deg):.10f}"
                out["override_applied"] = "1"
                applied += 1
                selected_rows.append(dict(out))
            else:
                out["suggested_alpha_deg"] = base_row["suggested_alpha_deg"]
                out["suggested_alpha_rad"] = base_row["suggested_alpha_rad"]
        else:
            for field in extra_fields:
                out.setdefault(field, "")

        full_rows.append(out)

    full_path = os.path.join(args.output_dir, args.full_name)
    with open(full_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row in full_rows:
            writer.writerow({k: row.get(k, "") for k in output_fields})

    selected_path = os.path.join(args.output_dir, args.selected_name)
    with open(selected_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({k: row.get(k, "") for k in output_fields})

    print(f"Saved full apply CSV: {full_path}")
    print(f"Saved selected apply CSV: {selected_path}")
    print(f"Rows in baseline: {len(base_rows)}")
    print(f"Rows selected/applied: {applied}")


if __name__ == "__main__":
    main()
