#!/usr/bin/env python3
"""
Fuse per-pair rotation and per-pair Q/K scale into GGUF Wq/Wk weights.

Expected CSV columns:
  - layer
  - head          (KV head index)
  - pair
  - suggested_alpha_rad
  - final_k_scale
  - final_q_scale

The script applies, for each selected pair p:
  K :  s_k * R(phi)
  Q :  s_q * R(phi)

with the same rotation angle on Q/K and inverse-compatible scales coming from
an offline calibration step.
"""

import argparse
import csv
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader
except ImportError:
    print("Error: gguf library not found. pip install gguf")
    sys.exit(1)


def load_transform_csv(csv_path):
    transforms = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["layer"])
            head = int(row["head"])
            pair = int(row["pair"])
            alpha = float(row["suggested_alpha_rad"])
            k_scale = float(row.get("final_k_scale", "1.0") or 1.0)
            q_scale = float(row.get("final_q_scale", "1.0") or 1.0)
            transforms[(layer, head)].append((pair, alpha, k_scale, q_scale))
    return dict(transforms)


def apply_transform_to_weight(weight_f32, layer_idx, transforms, head_dim, n_kv_heads, wtype):
    n_q_heads = weight_f32.shape[0] // head_dim
    n_group = n_q_heads // n_kv_heads if wtype == "q" else 1

    total_rows = 0
    for kv_h in range(n_kv_heads):
        key = (layer_idx, kv_h)
        if key not in transforms:
            continue
        pairs = transforms[key]

        if wtype == "k":
            heads_to_apply = [(kv_h, None)]
        else:
            heads_to_apply = [(kv_h * n_group + g, g) for g in range(n_group)]

        for h, _ in heads_to_apply:
            base = h * head_dim
            for pair_idx, alpha, k_scale, q_scale in pairs:
                d0 = base + 2 * pair_idx
                d1 = base + 2 * pair_idx + 1
                if d1 >= weight_f32.shape[0]:
                    continue

                scale = k_scale if wtype == "k" else q_scale
                c = math.cos(alpha)
                s = math.sin(alpha)

                row0 = weight_f32[d0, :].copy()
                row1 = weight_f32[d1, :].copy()

                weight_f32[d0, :] = scale * (c * row0 - s * row1)
                weight_f32[d1, :] = scale * (s * row0 + c * row1)
                total_rows += 1

    return total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Fuse per-pair rotation and scale from CSV into GGUF Wq/Wk weights"
    )
    parser.add_argument("input_gguf", help="Input F16 GGUF file")
    parser.add_argument("transform_csv", help="CSV with suggested_alpha_rad and final_{q,k}_scale")
    parser.add_argument("-o", "--output", required=True, help="Output GGUF file")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument("--q-only", action="store_true", help="Only fuse into Wq")
    parser.add_argument("--k-only", action="store_true", help="Only fuse into Wk")
    args = parser.parse_args()

    print("=" * 60)
    print("GGUF Per-Pair Rotation+Scale Fusion")
    print("=" * 60)

    print(f"\n[1/4] Loading transform CSV: {args.transform_csv}")
    transforms = load_transform_csv(args.transform_csv)
    n_entries = len(transforms)
    n_pairs = sum(len(v) for v in transforms.values())
    print(f"  (layer, kv_head) entries: {n_entries}")
    print(f"  Total rows: {n_pairs}")
    if transforms:
        sample_key = next(iter(transforms))
        sample_pair = transforms[sample_key][0]
        print(
            f"  Sample: L{sample_key[0]} H{sample_key[1]} P{sample_pair[0]} "
            f"alpha={math.degrees(sample_pair[1]):.2f}° k_scale={sample_pair[2]:.6f} q_scale={sample_pair[3]:.6f}"
        )

    print(f"\n[2/4] Copying {args.input_gguf} -> {args.output} ...")
    shutil.copy2(args.input_gguf, args.output)
    sz_gb = Path(args.input_gguf).stat().st_size / (1024 ** 3)
    print(f"  Copied {sz_gb:.2f} GB")

    print("\n[3/4] Reading tensor info ...")
    reader = GGUFReader(args.input_gguf)
    tensors_to_modify = []
    for tensor in reader.tensors:
        is_q = ".attn_q.weight" in tensor.name
        is_k = ".attn_k.weight" in tensor.name

        if args.q_only and is_k:
            continue
        if args.k_only and is_q:
            continue

        if is_q or is_k:
            parts = tensor.name.split(".")
            if len(parts) > 1 and parts[0] == "blk":
                layer_idx = int(parts[1])
                wtype = "q" if is_q else "k"
                tensors_to_modify.append(
                    {
                        "name": tensor.name,
                        "layer": layer_idx,
                        "type": wtype,
                        "offset": tensor.data_offset,
                        "shape": tensor.shape,
                        "original_data": tensor.data,
                    }
                )

    print(f"  Found {len(tensors_to_modify)} tensors to modify")

    k_tensors = [t for t in tensors_to_modify if t["type"] == "k"]
    if k_tensors:
        n_kv_heads = k_tensors[0]["original_data"].shape[0] // args.head_dim
    else:
        n_kv_heads = 8
    print(f"  n_kv_heads={n_kv_heads}, head_dim={args.head_dim}")

    print("\n[4/4] Applying transforms ...")
    total_fused = 0
    with open(args.output, "r+b") as f:
        for idx, info in enumerate(tensors_to_modify):
            weight = info["original_data"].astype(np.float32)
            n_rot = apply_transform_to_weight(
                weight,
                info["layer"],
                transforms,
                args.head_dim,
                n_kv_heads,
                info["type"],
            )
            f.seek(info["offset"])
            f.write(weight.astype(np.float16).tobytes())
            total_fused += n_rot

            if (idx + 1) % 10 == 0 or idx == len(tensors_to_modify) - 1:
                print(f"  {idx + 1}/{len(tensors_to_modify)} tensors, {total_fused} pair-transforms so far")

    print("\n" + "=" * 60)
    print(f"  Total pair-transforms applied: {total_fused}")
    print(f"  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
