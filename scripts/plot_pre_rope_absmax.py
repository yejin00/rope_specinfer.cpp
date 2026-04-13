#!/usr/bin/env python3
"""
Plot per-dimension Pre-RoPE Key AbsMax bar chart from rope_values_full.bin.

Usage:
    python scripts/plot_pre_rope_absmax.py <rope_values_full.bin> [--layer L] [--head H] [--output-dir DIR] [--max-tokens N]
"""

import struct
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MAGIC_ROPV = 0x524F5056  # "ROPV"


def read_pre_rope_absmax(file_path, target_layer, max_tokens=200000):
    """Read ROPV file and compute per-dim absmax of pre-RoPE values for one layer."""
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected ROPV")

        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads  = struct.unpack('I', f.read(4))[0]
        n_dims   = struct.unpack('I', f.read(4))[0]
        n_tokens = struct.unpack('I', f.read(4))[0]

        print(f"  layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")

        stride = n_heads * n_dims

        for layer in range(n_layers):
            # --- pre ---
            pre_count = struct.unpack('I', f.read(4))[0]
            if layer == target_layer:
                actual_tokens = pre_count // stride
                use_tokens = min(actual_tokens, max_tokens)
                read_floats = use_tokens * stride
                data = np.frombuffer(f.read(read_floats * 4), dtype=np.float32).copy()
                skip = pre_count - read_floats
                if skip > 0:
                    f.seek(skip * 4, 1)
                data = data.reshape(use_tokens, n_heads, n_dims)
                # absmax per head per dim
                absmax = np.max(np.abs(data), axis=0)  # (n_heads, n_dims)
                # skip post
                post_count = struct.unpack('I', f.read(4))[0]
                f.seek(post_count * 4, 1)
                return absmax, use_tokens, n_heads, n_dims
            else:
                f.seek(pre_count * 4, 1)

            # --- post ---
            post_count = struct.unpack('I', f.read(4))[0]
            f.seek(post_count * 4, 1)

    raise ValueError(f"Layer {target_layer} not found")


def plot_absmax(absmax_per_dim, layer, head, n_tokens, output_dir, n_dims,
                no_pair_highlight=False, highlight_start_pair=32):
    """Plot per-dimension absmax bar chart with block color regions and pair highlights."""
    dims = np.arange(n_dims)
    block_size = 32
    n_blocks = (n_dims + block_size - 1) // block_size

    fig, ax = plt.subplots(figsize=(16, 5))
    base_width = 0.65
    highlight_width = 0.85

    # Base bars (same style family as compare_absmax_rotation_nolabel_rpn_only.py)
    ax.bar(dims, absmax_per_dim, color='steelblue', alpha=0.6, width=base_width, edgecolor='none')

    # Late-pair highlight (pair p -> dims 2p, 2p+1), color palette matches compare script
    pair_dims = []
    if not no_pair_highlight:
        cmap = plt.cm.get_cmap('tab10')
        n_pairs = n_dims // 2
        start_pair = max(0, min(highlight_start_pair, n_pairs))
        for ci, pidx in enumerate(range(start_pair, n_pairs)):
            d0 = 2 * pidx
            d1 = 2 * pidx + 1
            pc = cmap(ci % 10)
            pair_dims.append((d0, d1, pidx, pc))

        for (d0, d1, pidx, pc) in pair_dims:
            for d in [d0, d1]:
                if d < n_dims:
                    ax.bar(d, absmax_per_dim[d], color=pc, alpha=0.85,
                           width=highlight_width, edgecolor='black', linewidth=0.8)

        # Pair connector arrows intentionally omitted.

    # Block shading (same palette as compare script)
    ax.axvspan(0, 32, alpha=0.08, color='red', label='Block 0 (0-31)')
    ax.axvspan(32, 64, alpha=0.08, color='orange', label='Block 1 (32-63)')
    ax.axvspan(64, 96, alpha=0.08, color='green', label='Block 2 (64-95)')
    ax.axvspan(96, 128, alpha=0.08, color='blue', label='Block 3 (96-127)')

    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('AbsMax', fontsize=12)
    ax.set_title(f'Pre-RoPE — (Layer {layer}, Head {head})',
                 fontsize=13, fontweight='bold')
    ax.set_xlim([-2, n_dims + 2])
    ax.grid(True, axis='y', alpha=0.3)

    # Legend (deduplicate)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, f'absmax_l{layer}_h{head}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Plot Pre-RoPE K AbsMax per dimension')
    parser.add_argument('rope_file', help='Path to rope_values_*_full.bin (ROPV format)')
    parser.add_argument('--layer', type=int, default=0, help='Layer index (default: 0)')
    parser.add_argument('--head', type=int, default=0, help='Head index (default: 0)')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    parser.add_argument('--max-tokens', type=int, default=200000, help='Max tokens to load')
    parser.add_argument('--no-pair-highlight', action='store_true',
                        help='Disable late pair color highlight')
    parser.add_argument('--highlight-start-pair', type=int, default=32,
                        help='Start pair index for color highlight (default: 32)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.rope_file) or '.'
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading {args.rope_file} ...")
    absmax, n_tokens, n_heads, n_dims = read_pre_rope_absmax(
        args.rope_file, args.layer, args.max_tokens)

    print(f"  Loaded layer {args.layer}: {n_tokens} tokens, {n_heads} heads, {n_dims} dims")
    print(f"  Plotting head {args.head} ...")
    plot_absmax(
        absmax[args.head], args.layer, args.head, n_tokens, args.output_dir, n_dims,
        no_pair_highlight=args.no_pair_highlight,
        highlight_start_pair=args.highlight_start_pair,
    )
    print("Done!")


if __name__ == '__main__':
    main()
