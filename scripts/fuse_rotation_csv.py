#!/usr/bin/env python3
"""
Fuse per-pair 2×2 rotation matrices into GGUF Wq/Wk weights (offline, in-place).

Reads a rotation CSV (from analyze_rope_rotation_late_angle.py) and applies
R(α) to each specified (layer, kv_head, pair) in the F16 GGUF file.

Weight layout (F16 GGUF, LLaMA):
    attn_k.weight: shape [in_features=hidden, out_features=n_kv_heads*head_dim]
    attn_q.weight: shape [in_features=hidden, out_features=n_heads*head_dim]

    For KV head h, RoPE pair p (llama.cpp non-interleaved convention):
        row_real = h * head_dim + 2*p
        row_imag = h * head_dim + 2*p + 1

    Rotation R(α) on pair p:
        new_real =  cos(α) * old_real - sin(α) * old_imag
        new_imag =  sin(α) * old_real + cos(α) * old_imag

    For Q with GQA (n_group = n_q_heads / n_kv_heads):
        each KV head maps to n_group Q heads → apply same rotation to all.

Usage:
    python fuse_rotation_csv.py /data/jongjip/models/llama3_8b.gguf \\
        /home/yjkim00/rope_specinfer.cpp/rope_rotation_shape_late_allheads/shape_orientation_table_filter.csv \\
        -o /data/jongjip/models/llama3_8b_rotated.gguf
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


# ── CSV loading ──────────────────────────────────────────────────────

def load_rotation_csv(csv_path):
    """Return {(layer, kv_head): [(pair_idx, alpha_rad), ...]}"""
    rotations = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["layer"])
            head = int(row["head"])
            pair = int(row["pair"])
            alpha = float(row["suggested_alpha_rad"])
            rotations[(layer, head)].append((pair, alpha))
    return dict(rotations)


# ── Rotation application ─────────────────────────────────────────────

def apply_rotation_to_weight(weight_f32, layer_idx, rotations, head_dim, n_kv_heads, wtype):
    """
    Apply per-pair 2×2 rotation to weight rows in-place.

    weight_f32: numpy float32 array, shape [out_features, in_features]
                data.shape = (n_heads*head_dim, hidden_size) for GGUF F16
    """
    n_q_heads = weight_f32.shape[0] // head_dim
    n_group = n_q_heads // n_kv_heads if wtype == "q" else 1

    total_rotated = 0
    for kv_h in range(n_kv_heads):
        key = (layer_idx, kv_h)
        if key not in rotations:
            continue
        pairs = rotations[key]

        heads_to_rotate = []
        if wtype == "k":
            heads_to_rotate = [kv_h]
        else:  # q — apply to all Q heads in the GQA group
            heads_to_rotate = [kv_h * n_group + g for g in range(n_group)]

        for h in heads_to_rotate:
            base = h * head_dim
            for pair_idx, alpha in pairs:
                d0 = base + 2 * pair_idx      # real part (row) — llama.cpp non-interleaved
                d1 = base + 2 * pair_idx + 1  # imag part (row) — adjacent dim

                if d0 >= weight_f32.shape[0] or d1 >= weight_f32.shape[0]:
                    continue

                c = math.cos(alpha)
                s = math.sin(alpha)

                # weight_f32 shape: [out_features, in_features]
                # rows d0, d1 are the two output dims of this pair
                row0 = weight_f32[d0, :].copy()
                row1 = weight_f32[d1, :].copy()

                weight_f32[d0, :] =  c * row0 - s * row1
                weight_f32[d1, :] =  s * row0 + c * row1

                total_rotated += 1

    return total_rotated


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fuse per-pair rotation from CSV into GGUF Wq/Wk weights"
    )
    parser.add_argument("input_gguf", help="Input F16 GGUF file")
    parser.add_argument("rotation_csv", help="Rotation CSV (shape_orientation_table_filter.csv)")
    parser.add_argument("-o", "--output", required=True, help="Output GGUF file")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension (default: 128)")
    parser.add_argument("--q-only", action="store_true", help="Only fuse into Wq (skip Wk)")
    parser.add_argument("--k-only", action="store_true", help="Only fuse into Wk (skip Wq)")
    args = parser.parse_args()

    print("=" * 60)
    print("GGUF Per-Pair Rotation Fusion")
    print("=" * 60)

    # 1. Load rotation CSV
    print(f"\n[1/4] Loading rotation CSV: {args.rotation_csv}")
    rotations = load_rotation_csv(args.rotation_csv)
    n_entries = len(rotations)
    n_pairs = sum(len(v) for v in rotations.values())
    layers_touched = len(set(k[0] for k in rotations))
    print(f"  (layer, kv_head) entries: {n_entries}")
    print(f"  Total pairs: {n_pairs}")
    print(f"  Layers: {layers_touched}")

    # Sample
    sample_key = next(iter(rotations))
    sample_pairs = rotations[sample_key]
    print(f"  Sample: L{sample_key[0]} H{sample_key[1]}: {len(sample_pairs)} pairs, "
          f"first pair={sample_pairs[0][0]} α={math.degrees(sample_pairs[0][1]):.2f}°")

    # 2. Copy GGUF
    print(f"\n[2/4] Copying {args.input_gguf} → {args.output} ...")
    shutil.copy2(args.input_gguf, args.output)
    sz_gb = Path(args.input_gguf).stat().st_size / (1024**3)
    print(f"  Copied {sz_gb:.2f} GB")

    # 3. Read tensor info
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
                tensors_to_modify.append({
                    "name": tensor.name,
                    "layer": layer_idx,
                    "type": wtype,
                    "offset": tensor.data_offset,
                    "shape": tensor.shape,
                    "original_data": tensor.data,
                })

    print(f"  Found {len(tensors_to_modify)} tensors to modify")

    # Infer n_kv_heads from K tensor data shape (data.shape[0] = out_features)
    k_tensors = [t for t in tensors_to_modify if t["type"] == "k"]
    if k_tensors:
        n_kv_heads = k_tensors[0]["original_data"].shape[0] // args.head_dim
    else:
        q_tensors = [t for t in tensors_to_modify if t["type"] == "q"]
        # fallback: guess from Q (might be wrong for GQA, but K-only won't hit this)
        n_kv_heads = 8  # LLaMA-3-8B default
    print(f"  n_kv_heads={n_kv_heads}, head_dim={args.head_dim}")

    # 4. Apply rotation and write
    print("\n[4/4] Applying rotation ...")
    total_fused = 0
    with open(args.output, "r+b") as f:
        for i, info in enumerate(tensors_to_modify):
            weight = info["original_data"].astype(np.float32)

            n_rot = apply_rotation_to_weight(
                weight, info["layer"], rotations, args.head_dim, n_kv_heads, info["type"]
            )

            weight_f16 = weight.astype(np.float16)

            f.seek(info["offset"])
            f.write(weight_f16.tobytes())

            total_fused += n_rot
            if (i + 1) % 10 == 0 or i == len(tensors_to_modify) - 1:
                print(f"  {i+1}/{len(tensors_to_modify)} tensors, {total_fused} pair-rotations so far")

    print(f"\n{'=' * 60}")
    print(f"  Total pair-rotations applied: {total_fused}")
    print(f"  Output: {args.output}")
    print(f"{'=' * 60}")

    print(f"\nNext steps:")
    print(f"  # Quantize to Q4_0:")
    print(f"  ./build/bin/llama-quantize {args.output} {args.output.replace('.gguf', '_q4_0.gguf')} q4_0")
    print(f"")
    print(f"  # Or run perplexity directly on F16:")
    print(f"  ./build/bin/llama-perplexity -m {args.output} -f wikitext-2-raw/wiki.test.raw")


if __name__ == "__main__":
    main()
