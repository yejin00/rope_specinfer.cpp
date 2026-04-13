#!/usr/bin/env python3
"""
Filter a rotation CSV by head-wise top-k late setter pairs.

Input 1: full rotation CSV, e.g. shape_orientation_table_post_noclip.csv
Input 2: scale_setter_summary_post.csv from analyze_head_scale_setters.py

For each (layer, head), parse the top_pairs field from the setter summary, keep the
first top-k pairs whose pair index >= pair_start, and emit only those rows from the
rotation CSV.
"""

import argparse
import csv
from collections import defaultdict


def parse_top_pairs(text):
    out = []
    for item in (text or "").split(";"):
        item = item.strip()
        if not item:
            continue
        pair_s, count_s = item.split(":", 1)
        out.append((int(pair_s), int(count_s)))
    return out


def load_rotation_rows(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = []
        for row in reader:
            row["_layer"] = int(row["layer"])
            row["_head"] = int(row["head"])
            row["_pair"] = int(row["pair"])
            rows.append(row)
    return fieldnames, rows


def build_topk_map(summary_csv, pair_start, top_k):
    selected = {}
    with open(summary_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["layer"])
            head = int(row["head"])
            top_pairs = parse_top_pairs(row.get("top_pairs", ""))
            keep = []
            for pair, _count in top_pairs:
                if pair < pair_start:
                    continue
                if pair not in keep:
                    keep.append(pair)
                if len(keep) >= top_k:
                    break
            selected[(layer, head)] = keep
    return selected


def main():
    parser = argparse.ArgumentParser(description="Filter rotation CSV by head-wise top-k late setter pairs")
    parser.add_argument("input_csv", help="Full rotation CSV")
    parser.add_argument("setter_summary_csv", help="scale_setter_summary_post.csv or pre.csv")
    parser.add_argument("output_csv", help="Filtered output CSV")
    parser.add_argument("--pair-start", type=int, default=32, help="Late-pair start index (default: 32)")
    parser.add_argument("--top-k", type=int, default=4, help="Keep top-k late setter pairs per head (default: 4)")
    parser.add_argument("--analysis-csv", help="Optional analysis CSV with one row per head and selected pairs")
    args = parser.parse_args()

    fieldnames, rows = load_rotation_rows(args.input_csv)
    selected = build_topk_map(args.setter_summary_csv, args.pair_start, args.top_k)

    kept_rows = []
    analysis_rows = []
    counts = defaultdict(int)
    for row in rows:
        key = (row["_layer"], row["_head"])
        keep_pairs = selected.get(key, [])
        if row["_pair"] in keep_pairs:
            kept_rows.append(row)
            counts[key] += 1

    for key in sorted(selected.keys()):
        keep_pairs = selected[key]
        analysis_rows.append(
            {
                "layer": key[0],
                "head": key[1],
                "selected_pair_count": len(keep_pairs),
                "selected_pairs": ";".join(str(x) for x in keep_pairs),
                "rows_emitted": counts[key],
            }
        )

    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in kept_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    if args.analysis_csv:
        with open(args.analysis_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["layer", "head", "selected_pair_count", "selected_pairs", "rows_emitted"],
            )
            writer.writeheader()
            writer.writerows(analysis_rows)

    nonempty_heads = sum(1 for pairs in selected.values() if pairs)
    print(f"Saved filtered CSV: {args.output_csv}")
    print(f"Kept rows: {len(kept_rows)}")
    print(f"Heads with >=1 selected late setter pair: {nonempty_heads} / {len(selected)}")
    if args.analysis_csv:
        print(f"Saved analysis CSV: {args.analysis_csv}")


if __name__ == "__main__":
    main()
