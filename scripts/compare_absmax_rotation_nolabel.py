#!/usr/bin/env python3
"""
Compare AbsMax (APOR format) between Original and Rotation-fused models.
Highlights rotated pair dims from CSV instead of CRS scales.
"""

import struct
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from collections import defaultdict

MAGIC_APOR = 0x524F5041  # "APOR"
MAGIC_ROPV = 0x524F5056  # "ROPV"


def load_absmax(file_path):
    """Load AbsMax from .bin file (APOR or ROPV format)"""
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_APOR and magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected APOR or ROPV")

        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]

        print(f"  Header: layers={n_layers}, heads={n_heads}, dims={n_dims}")

        pre_absmax = []
        post_absmax = []

        for layer in range(n_layers):
            layer_pre = []
            layer_post = []
            for head in range(n_heads):
                data_pre = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                layer_pre.append(data_pre)
                data_post = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                layer_post.append(data_post)

            pre_absmax.append(np.array(layer_pre))
            post_absmax.append(np.array(layer_post))

        return {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'pre_absmax': pre_absmax,
            'post_absmax': post_absmax
        }


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


def select_absmax_slot(data, layer_idx, head_idx, use_post_rope, label):
    """Select pre/post absmax slot, with fallback if the requested post slot is empty."""
    if use_post_rope:
        vals = data['post_absmax'][layer_idx][head_idx]
        if vals.max() == 0:
            print(f"  ⚠ {label} post slot is empty, falling back to pre slot")
            vals = data['pre_absmax'][layer_idx][head_idx]
        return vals

    return data['pre_absmax'][layer_idx][head_idx]


def visualize_comparison(orig_absmax, rot_absmax, rotation_pairs, layer_idx, head_idx,
                         use_post_rope, head_dim, output_path=None):
    """
    rotation_pairs: list of (pair_idx, alpha_rad) for this (layer, head)
    """
    n_dims = len(orig_absmax)
    dims = np.arange(n_dims)

    rope_stage = "Post-RoPE" if use_post_rope else "Pre-RoPE"

    # Pair color palette
    cmap = plt.cm.get_cmap('tab10')

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    base_width = 0.65
    highlight_width = 0.85

    # Collect pair dims for highlighting (llama.cpp non-interleaved: pair p → dims 2p, 2p+1)
    pair_dims = []  # (d0, d1, pair_idx, alpha_deg, color)
    for ci, (pidx, alpha) in enumerate(rotation_pairs):
        d0 = 2 * pidx
        d1 = 2 * pidx + 1
        pc = cmap(ci % 10)
        pair_dims.append((d0, d1, pidx, math.degrees(alpha), pc))

    for ax_idx, (ax, absmax, title_prefix, base_color) in enumerate([
        (axes[0], orig_absmax, 'Original Model', 'steelblue'),
        (axes[1], rot_absmax, 'Rotation-Fused Model', 'forestgreen'),
    ]):
        # Base bars
        ax.bar(dims, absmax, color=base_color, alpha=0.6, width=base_width, edgecolor='none')

        # Highlight rotated pair dims with per-pair colors
        for (hf_d0, hf_d1, pidx, adeg, pc) in pair_dims:
            for hf_d in [hf_d0, hf_d1]:
                if hf_d < n_dims:
                    ax.bar(hf_d, absmax[hf_d], color=pc, alpha=0.85,
                           width=highlight_width, edgecolor='black', linewidth=0.8)

        # Annotate pair info
        used_positions = {}
        for (hf_d0, hf_d1, pidx, adeg, pc) in pair_dims:
            if hf_d0 >= n_dims:
                continue

            y0 = absmax[hf_d0]
            y1 = absmax[hf_d1] if hf_d1 < n_dims else 0

            # Label at hf_d0
                        # Pair text labels removed; keep per-pair colors and connector arrows only.
            used_positions[hf_d0] = used_positions.get(hf_d0, 0) + 1

            # Connect the pair dims with a thin line
            if hf_d1 < n_dims:
                ax.annotate('', xy=(hf_d1, y1), xytext=(hf_d0, y0),
                            arrowprops=dict(arrowstyle='->', color=pc, lw=1.5, alpha=0.7))

        # Block shading
        ax.axvspan(0, 32, alpha=0.08, color='red', label='Block 0 (0-31)')
        ax.axvspan(32, 64, alpha=0.08, color='orange', label='Block 1 (32-63)')
        ax.axvspan(64, 96, alpha=0.08, color='green', label='Block 2 (64-95)')
        ax.axvspan(96, 128, alpha=0.08, color='blue', label='Block 3 (96-127)')

        ax.set_title(f'{title_prefix}\nLayer {layer_idx}, Head {head_idx} ({rope_stage})',
                     fontsize=12, weight='bold')
        ax.set_xlim([-2, n_dims + 2])
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    # Sync y-limits
    y_max = max(np.max(orig_absmax), np.max(rot_absmax)) * 1.2
    axes[0].set_ylim([0, y_max])
    axes[1].set_ylim([0, y_max])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved: {output_path}")
    else:
        plt.show()

    # Print statistics
    print(f"\n{'='*70}")
    print(f"Rotation Comparison: Layer {layer_idx}, Head {head_idx} ({rope_stage})")
    print(f"{'='*70}")
    print(f"{'Pair':<6} {'Dim0':<6} {'Dim1':<6} {'Angle':<10} {'Orig_d0':<10} {'Rot_d0':<10} {'Orig_d1':<10} {'Rot_d1':<10}")
    print("-" * 70)

    for (hf_d0, hf_d1, pidx, adeg, pc) in pair_dims:
        o0 = orig_absmax[hf_d0] if hf_d0 < n_dims else 0
        r0 = rot_absmax[hf_d0] if hf_d0 < n_dims else 0
        o1 = orig_absmax[hf_d1] if hf_d1 < n_dims else 0
        r1 = rot_absmax[hf_d1] if hf_d1 < n_dims else 0
        print(f"p{pidx:<5} {hf_d0:<6} {hf_d1:<6} {adeg:<10.2f} {o0:<10.4f} {r0:<10.4f} {o1:<10.4f} {r1:<10.4f}")

    # Overall stats
    print(f"\nOverall: orig max={np.max(orig_absmax):.4f} mean={np.mean(orig_absmax):.4f}")
    print(f"         rot  max={np.max(rot_absmax):.4f} mean={np.mean(rot_absmax):.4f}")

    # Pair imbalance ratio
    print(f"\nPair imbalance (|d0-d1| / max(d0,d1)):")
    for (hf_d0, hf_d1, pidx, adeg, pc) in pair_dims:
        o0, o1 = orig_absmax[hf_d0], orig_absmax[hf_d1]
        r0, r1 = rot_absmax[hf_d0], rot_absmax[hf_d1]
        imb_orig = abs(o0 - o1) / max(o0, o1, 1e-8)
        imb_rot = abs(r0 - r1) / max(r0, r1, 1e-8)
        print(f"  p{pidx}: orig={imb_orig:.3f} → rot={imb_rot:.3f}  {'✓ improved' if imb_rot < imb_orig else '✗ worse'}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare AbsMax: Original vs Rotation-Fused'
    )
    parser.add_argument('orig_absmax', help='Original model absmax .bin (APOR)')
    parser.add_argument('rot_absmax', help='Rotation-fused model absmax .bin (APOR)')
    parser.add_argument('rotation_csv', nargs='?', default=None,
                        help='Optional rotation CSV file for highlighting rotated pairs')
    parser.add_argument('--layer', type=int, default=0, help='Layer index')
    parser.add_argument('--head', type=int, default=0, help='Head index (KV head)')
    parser.add_argument('--post-rope', action='store_true',
                        help='Read both original and rotated absmax from post slot (fallback to pre if empty)')
    parser.add_argument('--head-dim', type=int, default=128, help='Head dimension')
    parser.add_argument('-o', '--output', help='Output image path')

    args = parser.parse_args()

    print("=" * 70)
    print("AbsMax Comparison: Original vs Rotation-Fused")
    print("=" * 70)

    print(f"\nLoading {args.orig_absmax} ...")
    orig_data = load_absmax(args.orig_absmax)

    print(f"Loading {args.rot_absmax} ...")
    rot_data = load_absmax(args.rot_absmax)

    rotations = {}
    if args.rotation_csv:
        print(f"Loading {args.rotation_csv} ...")
        rotations = load_rotation_csv(args.rotation_csv)
    else:
        print("No rotation CSV provided; comparing absmax values without pair highlights")

    layer_idx = args.layer
    head_idx = args.head

    orig_vals = select_absmax_slot(orig_data, layer_idx, head_idx, args.post_rope, "Original")
    rot_vals = select_absmax_slot(rot_data, layer_idx, head_idx, args.post_rope, "Rotated")

    pairs = rotations.get((layer_idx, head_idx), [])
    if not pairs:
        print(f"\n⚠ No rotation pairs for Layer {layer_idx}, Head {head_idx}")
        print(f"  Available (layer, head) keys: {sorted(rotations.keys())[:20]}...")

    visualize_comparison(
        orig_vals, rot_vals, pairs, layer_idx, head_idx,
        args.post_rope, args.head_dim, args.output
    )


if __name__ == '__main__':
    main()
