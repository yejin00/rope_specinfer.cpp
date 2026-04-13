#!/usr/bin/env python3
"""
Plot per-dimension K AbsMax from APOR file (rope_absmax_*.bin).

Usage:
    python scripts/plot_apor_absmax.py <rope_absmax.bin> --layer 0 --head 0 --output-dir dump_absmax --post-rope
"""

import argparse
import os
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MAGIC_APOR = 0x524F5041  # "APOR"
MAGIC_ROPV = 0x524F5056  # "ROPV"


def read_apor(file_path):
    with open(file_path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic not in (MAGIC_APOR, MAGIC_ROPV):
            raise ValueError(f"Invalid magic: {hex(magic)}, expected APOR or ROPV")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]

        pre_absmax = []
        post_absmax = []
        for _ in range(n_layers):
            layer_pre = []
            layer_post = []
            for _ in range(n_heads):
                layer_pre.append(np.frombuffer(f.read(n_dims * 4), dtype=np.float32).copy())
                layer_post.append(np.frombuffer(f.read(n_dims * 4), dtype=np.float32).copy())
            pre_absmax.append(np.array(layer_pre))
            post_absmax.append(np.array(layer_post))

    return {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_dims": n_dims,
        "pre_absmax": pre_absmax,
        "post_absmax": post_absmax,
    }


def plot_absmax(absmax_per_dim, layer, head, output_dir, n_dims, rope_stage,
                pair_highlight=False, highlight_start_pair=32):
    dims = np.arange(n_dims)
    fig, ax = plt.subplots(figsize=(16, 5))
    base_width = 0.65
    highlight_width = 0.85

    ax.bar(dims, absmax_per_dim, color="steelblue", alpha=0.6, width=base_width, edgecolor="none")

    if pair_highlight:
        cmap = plt.cm.get_cmap("tab10")
        n_pairs = n_dims // 2
        start_pair = max(0, min(highlight_start_pair, n_pairs))
        for ci, pidx in enumerate(range(start_pair, n_pairs)):
            d0 = 2 * pidx
            d1 = 2 * pidx + 1
            pc = cmap(ci % 10)
            for d in (d0, d1):
                if d < n_dims:
                    ax.bar(d, absmax_per_dim[d], color=pc, alpha=0.85,
                           width=highlight_width, edgecolor="black", linewidth=0.8)

    ax.axvspan(0, 32, alpha=0.08, color="red", label="Block 0 (0-31)")
    ax.axvspan(32, 64, alpha=0.08, color="orange", label="Block 1 (32-63)")
    ax.axvspan(64, 96, alpha=0.08, color="green", label="Block 2 (64-95)")
    ax.axvspan(96, 128, alpha=0.08, color="blue", label="Block 3 (96-127)")

    ax.set_xlabel("Dimension", fontsize=12)
    ax.set_ylabel("AbsMax", fontsize=12)
    ax.set_title(f"Post-RoPE — Layer {layer}, Head {head}",
                 fontsize=13, fontweight="bold")
    ax.set_xlim([-2, n_dims + 2])
    ax.grid(True, axis="y", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9)

    plt.tight_layout()
    stage_name = rope_stage.lower().replace("-", "_")
    out_path = os.path.join(output_dir, f"apor_absmax_{stage_name}_l{layer}_h{head}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot APOR K AbsMax per dimension")
    parser.add_argument("apor_file", help="Path to rope_absmax_*.bin (APOR format)")
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    parser.add_argument("--head", type=int, default=0, help="Head index")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--post-rope", action="store_true",
                        help="Use post slot; if empty, fallback to pre slot")
    parser.add_argument("--pair-highlight", action="store_true",
                        help="Enable late pair color highlight (default: off)")
    parser.add_argument("--highlight-start-pair", type=int, default=32,
                        help="Start pair index for color highlight (default: 32)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.apor_file) or "."
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading {args.apor_file} ...")
    data = read_apor(args.apor_file)
    print(f"  layers={data['n_layers']}, heads={data['n_heads']}, dims={data['n_dims']}")

    if args.layer < 0 or args.layer >= data["n_layers"]:
        raise ValueError(f"layer out of range: {args.layer}")
    if args.head < 0 or args.head >= data["n_heads"]:
        raise ValueError(f"head out of range: {args.head}")

    if args.post_rope:
        vals = data["post_absmax"][args.layer][args.head]
        rope_stage = "Post-RoPE"
        if np.max(vals) == 0:
            print("  ⚠ Post slot is empty, falling back to pre slot")
            vals = data["pre_absmax"][args.layer][args.head]
            rope_stage = "Pre-RoPE (fallback)"
    else:
        vals = data["pre_absmax"][args.layer][args.head]
        rope_stage = "Pre-RoPE"

    plot_absmax(
        vals, args.layer, args.head, args.output_dir, data["n_dims"], rope_stage,
        pair_highlight=args.pair_highlight,
        highlight_start_pair=args.highlight_start_pair,
    )
    print("Done!")


if __name__ == "__main__":
    main()
