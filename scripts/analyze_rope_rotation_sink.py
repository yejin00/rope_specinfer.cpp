#!/usr/bin/env python3
"""
RoPE Rotation Angle Analysis Script

Analyzes pre-RoPE and post-RoPE key values to understand:
1. Rotation angle per pair index (deterministic from freq_base)
2. Pair scatter plots: (x_pre, y_pre) vs (x_post, y_post)
3. Pair imbalance: |x|/(|x|+|y|) distribution before/after RoPE
4. Token-wise imbalance change for selected pairs

Usage:
    python analyze_rope_rotation.py <rope_values.bin> [--output-dir <dir>]
"""

import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

MAGIC_ROPE = 0x524F5056  # "ROPV"


def read_chunk(f, n_floats):
    """Read n_floats float32 values, handling partial reads."""
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        print(f"  [warn] Expected {n_floats} floats, got {got}")
    return np.frombuffer(raw[:got * 4], dtype=np.float32).copy()


def read_rope_values(file_path, max_tokens=10000):
    """Read rope_values.bin, limiting tokens for memory efficiency."""
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_ROPE:
            raise ValueError(f"Invalid magic: {hex(magic)}")

        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        n_tokens = struct.unpack('I', f.read(4))[0]
        stride = n_heads * n_dims

        print(f"File: {file_path}")
        print(f"  Version={version}, Layers={n_layers}, Heads={n_heads}, Dims={n_dims}, Tokens={n_tokens}")

        layer_data = []
        for layer in range(n_layers):
            # --- pre ---
            hdr = f.read(4)
            if len(hdr) < 4:
                print(f"  [warn] EOF at layer {layer} pre_count")
                layer_data.append({'pre': None, 'post': None})
                continue
            pre_count = struct.unpack('I', hdr)[0]

            pre_rope = None
            if pre_count > 0:
                actual_tokens_pre = pre_count // stride
                use_pre = min(actual_tokens_pre, max_tokens)
                # Read only what we need, skip the rest
                read_floats = use_pre * stride
                skip_floats = pre_count - read_floats
                pre_rope = read_chunk(f, read_floats)
                if skip_floats > 0:
                    f.seek(skip_floats * 4, 1)
                if len(pre_rope) == read_floats:
                    pre_rope = pre_rope.reshape(use_pre, n_heads, n_dims)
                else:
                    pre_rope = None

            # --- post ---
            hdr = f.read(4)
            if len(hdr) < 4:
                print(f"  [warn] EOF at layer {layer} post_count")
                layer_data.append({'pre': pre_rope, 'post': None})
                continue
            post_count = struct.unpack('I', hdr)[0]

            post_rope = None
            if post_count > 0:
                actual_tokens_post = post_count // stride
                use_post = min(actual_tokens_post, max_tokens)
                read_floats = use_post * stride
                skip_floats = post_count - read_floats
                post_rope = read_chunk(f, read_floats)
                if skip_floats > 0:
                    f.seek(skip_floats * 4, 1)
                if len(post_rope) == read_floats:
                    post_rope = post_rope.reshape(use_post, n_heads, n_dims)
                else:
                    post_rope = None

            layer_data.append({'pre': pre_rope, 'post': post_rope})

            if layer == 0:
                pre_tok = pre_rope.shape[0] if pre_rope is not None else 0
                post_tok = post_rope.shape[0] if post_rope is not None else 0
                print(f"  Layer 0: pre_count={pre_count} → {pre_tok} tokens loaded, "
                      f"post_count={post_count} → {post_tok} tokens loaded")
                print(f"  (max_tokens={max_tokens})")

        return {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'n_tokens': n_tokens,
            'layers': layer_data
        }


def compute_rope_freqs(n_dims, freq_base=500000.0):
    """Compute RoPE frequency for each pair index."""
    n_pairs = n_dims // 2
    freqs = np.array([freq_base ** (-2.0 * i / n_dims) for i in range(n_pairs)])
    return freqs


def plot1_rotation_angle_vs_pair(n_dims, freq_base, output_dir):
    """Plot 1: Rotation angle per pair for different token positions."""
    freqs = compute_rope_freqs(n_dims, freq_base)
    n_pairs = n_dims // 2
    positions = [1, 10, 100, 500, 1000, 2000]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: absolute angle (radians)
    for pos in positions:
        angles = pos * freqs
        axes[0].plot(range(n_pairs), angles, label=f'pos={pos}', alpha=0.8)
    axes[0].set_xlabel('Pair Index')
    axes[0].set_ylabel('Rotation Angle (radians)')
    axes[0].set_title('RoPE Rotation Angle vs Pair Index')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Right: angle mod 2π (effective rotation)
    for pos in positions:
        angles = (pos * freqs) % (2 * np.pi)
        axes[1].scatter(range(n_pairs), angles, label=f'pos={pos}', alpha=0.6, s=10)
    axes[1].set_xlabel('Pair Index')
    axes[1].set_ylabel('Effective Rotation (rad, mod 2π)')
    axes[1].set_title('Effective Rotation Angle (mod 2π)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=np.pi, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'plot1_rotation_angle_vs_pair.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot2_pair_scatter(data, layers, heads, pairs, output_dir):
    """Plot 2: Scatter of (x,y) pairs pre vs post RoPE."""
    n_pairs_to_show = len(pairs)
    n_layers_to_show = len(layers)

    for head_idx in heads:
        fig, axes = plt.subplots(n_layers_to_show, n_pairs_to_show * 2, figsize=(5 * n_pairs_to_show * 2, 5 * n_layers_to_show))
        if n_layers_to_show == 1:
            axes = axes[np.newaxis, :]

        for li, layer_idx in enumerate(layers):
            pre = data['layers'][layer_idx]['pre']
            post = data['layers'][layer_idx]['post']
            if pre is None or post is None:
                continue

            n_tokens_use = min(pre.shape[0], post.shape[0], 5000)

            for pi, pair_idx in enumerate(pairs):
                d0 = pair_idx * 2
                d1 = pair_idx * 2 + 1

                # Pre-RoPE scatter
                x_pre = pre[:n_tokens_use, head_idx, d0]
                y_pre = pre[:n_tokens_use, head_idx, d1]
                ax = axes[li, pi * 2]
                ax.scatter(x_pre, y_pre, s=1, c='blue')
                ax.set_title(f'Pre-RoPE L{layer_idx} H{head_idx} P{pair_idx}', fontsize=9)
                ax.set_xlabel(f'dim {d0}')
                ax.set_ylabel(f'dim {d1}')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                lim = max(np.abs(x_pre).max(), np.abs(y_pre).max()) * 1.1
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.axhline(0, color='gray', lw=0.5)
                ax.axvline(0, color='gray', lw=0.5)

                # Post-RoPE scatter
                x_post = post[:n_tokens_use, head_idx, d0]
                y_post = post[:n_tokens_use, head_idx, d1]
                ax = axes[li, pi * 2 + 1]
                ax.scatter(x_post, y_post, s=1, c='red')
                ax.set_title(f'Post-RoPE L{layer_idx} H{head_idx} P{pair_idx}', fontsize=9)
                ax.set_xlabel(f'dim {d0}')
                ax.set_ylabel(f'dim {d1}')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                lim_post = max(np.abs(x_post).max(), np.abs(y_post).max()) * 1.1
                ax.set_xlim(-lim_post, lim_post)
                ax.set_ylim(-lim_post, lim_post)
                ax.axhline(0, color='gray', lw=0.5)
                ax.axvline(0, color='gray', lw=0.5)

        plt.suptitle(f'Pair Scatter: Pre-RoPE (blue) vs Post-RoPE (red) — Head {head_idx}', fontsize=14)
        plt.tight_layout()
        path = os.path.join(output_dir, f'plot2_pair_scatter_h{head_idx}.png')
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Saved: {path}")


def plot3_imbalance_histogram(data, layers, heads, pairs, output_dir):
    """Plot 3: Pair imbalance |x|/(|x|+|y|) histogram, pre vs post."""
    for head_idx in heads:
        fig, axes = plt.subplots(len(layers), len(pairs), figsize=(4 * len(pairs), 4 * len(layers)))
        if len(layers) == 1:
            axes = axes[np.newaxis, :]
        if len(pairs) == 1:
            axes = axes[:, np.newaxis]

        for li, layer_idx in enumerate(layers):
            pre = data['layers'][layer_idx]['pre']
            post = data['layers'][layer_idx]['post']
            if pre is None or post is None:
                continue

            n_tokens_use = min(pre.shape[0], post.shape[0])

            for pi, pair_idx in enumerate(pairs):
                d0 = pair_idx * 2
                d1 = pair_idx * 2 + 1

                x_pre = np.abs(pre[:n_tokens_use, head_idx, d0])
                y_pre = np.abs(pre[:n_tokens_use, head_idx, d1])
                imb_pre = x_pre / (x_pre + y_pre + 1e-10)

                x_post = np.abs(post[:n_tokens_use, head_idx, d0])
                y_post = np.abs(post[:n_tokens_use, head_idx, d1])
                imb_post = x_post / (x_post + y_post + 1e-10)

                ax = axes[li, pi]
                ax.hist(imb_pre, bins=50, alpha=0.5, label='Pre-RoPE', color='blue', density=True)
                ax.hist(imb_post, bins=50, alpha=0.5, label='Post-RoPE', color='red', density=True)
                ax.axvline(0.5, color='green', linestyle='--', lw=1, label='Balanced')
                ax.set_title(f'L{layer_idx} P{pair_idx} (dim {d0},{d1})', fontsize=9)
                ax.set_xlabel('|x| / (|x|+|y|)')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

        plt.suptitle(f'Pair Imbalance Distribution — Head {head_idx}\n'
                     f'(0.5 = balanced, 0 or 1 = one-sided)', fontsize=12)
        plt.tight_layout()
        path = os.path.join(output_dir, f'plot3_imbalance_hist_h{head_idx}.png')
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Saved: {path}")


def plot4_tokenwise_imbalance(data, layers, heads, pairs, output_dir, token_range=2000):
    """Plot 4: Token-wise imbalance change for selected pairs."""
    for head_idx in heads:
        fig, axes = plt.subplots(len(pairs), len(layers), figsize=(6 * len(layers), 3 * len(pairs)))
        if len(layers) == 1:
            axes = axes[:, np.newaxis] if len(pairs) > 1 else axes[np.newaxis, np.newaxis]
        if len(pairs) == 1:
            axes = axes[np.newaxis, :]

        for pi, pair_idx in enumerate(pairs):
            for li, layer_idx in enumerate(layers):
                pre = data['layers'][layer_idx]['pre']
                post = data['layers'][layer_idx]['post']
                if pre is None or post is None:
                    continue

                n_use = min(pre.shape[0], post.shape[0], token_range)
                d0 = pair_idx * 2
                d1 = pair_idx * 2 + 1

                x_pre = np.abs(pre[:n_use, head_idx, d0])
                y_pre = np.abs(pre[:n_use, head_idx, d1])
                imb_pre = x_pre / (x_pre + y_pre + 1e-10)

                x_post = np.abs(post[:n_use, head_idx, d0])
                y_post = np.abs(post[:n_use, head_idx, d1])
                imb_post = x_post / (x_post + y_post + 1e-10)

                ax = axes[pi, li]
                # Moving average for readability
                window = 50
                if n_use > window:
                    imb_pre_ma = np.convolve(imb_pre, np.ones(window)/window, mode='valid')
                    imb_post_ma = np.convolve(imb_post, np.ones(window)/window, mode='valid')
                    x_axis = np.arange(len(imb_pre_ma))
                    ax.plot(x_axis, imb_pre_ma, alpha=0.7, label='Pre-RoPE', color='blue', lw=0.8)
                    ax.plot(x_axis, imb_post_ma, alpha=0.7, label='Post-RoPE', color='red', lw=0.8)
                else:
                    ax.plot(imb_pre, alpha=0.5, label='Pre-RoPE', color='blue', lw=0.5)
                    ax.plot(imb_post, alpha=0.5, label='Post-RoPE', color='red', lw=0.5)

                ax.axhline(0.5, color='green', linestyle='--', lw=0.8)
                ax.set_title(f'L{layer_idx} P{pair_idx} (dim {d0},{d1})', fontsize=9)
                ax.set_xlabel('Token position')
                ax.set_ylabel('|x|/(|x|+|y|)')
                ax.set_ylim(0, 1)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

        plt.suptitle(f'Token-wise Pair Imbalance (MA={window}) — Head {head_idx}', fontsize=12)
        plt.tight_layout()
        path = os.path.join(output_dir, f'plot4_tokenwise_imbalance_h{head_idx}.png')
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Saved: {path}")


def plot5_rotation_direction(data, layers, heads, pairs, freq_base, output_dir):
    """Plot 5: Analyze rotation direction — classify movements as x-axis, y-axis, or diagonal."""
    n_dims = data['n_dims']
    freqs = compute_rope_freqs(n_dims, freq_base)

    for head_idx in heads:
        fig, axes = plt.subplots(len(layers), len(pairs), figsize=(5 * len(pairs), 5 * len(layers)))
        if len(layers) == 1:
            axes = axes[np.newaxis, :]
        if len(pairs) == 1:
            axes = axes[:, np.newaxis]

        for li, layer_idx in enumerate(layers):
            pre = data['layers'][layer_idx]['pre']
            post = data['layers'][layer_idx]['post']
            if pre is None or post is None:
                continue

            n_use = min(pre.shape[0], post.shape[0], 5000)

            for pi, pair_idx in enumerate(pairs):
                d0 = pair_idx * 2
                d1 = pair_idx * 2 + 1

                x_pre = pre[:n_use, head_idx, d0]
                y_pre = pre[:n_use, head_idx, d1]
                x_post = post[:n_use, head_idx, d0]
                y_post = post[:n_use, head_idx, d1]

                # Compute movement vector
                dx = x_post - x_pre
                dy = y_post - y_pre

                # Compute angle of movement
                move_angle = np.arctan2(dy, dx)  # [-π, π]

                ax = axes[li, pi]
                ax.hist(move_angle, bins=72, density=True, alpha=0.7, color='purple')
                ax.set_title(f'L{layer_idx} P{pair_idx} (dim {d0},{d1})', fontsize=9)
                ax.set_xlabel('Movement direction (rad)')
                ax.set_ylabel('Density')
                # Mark x-axis, y-axis, diagonal directions
                ax.axvline(0, color='red', lw=1, alpha=0.5, label='→ +x')
                ax.axvline(np.pi/2, color='blue', lw=1, alpha=0.5, label='↑ +y')
                ax.axvline(np.pi/4, color='green', lw=1, alpha=0.5, label='↗ diag')
                ax.axvline(-np.pi/4, color='orange', lw=1, alpha=0.5, label='↘ diag')
                ax.legend(fontsize=6)
                ax.grid(True, alpha=0.3)

        plt.suptitle(f'RoPE Movement Direction Distribution — Head {head_idx}\n'
                     f'(How each pair moves from pre→post RoPE)', fontsize=12)
        plt.tight_layout()
        path = os.path.join(output_dir, f'plot5_rotation_direction_h{head_idx}.png')
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='RoPE Rotation Angle Analysis')
    parser.add_argument('rope_file', help='Path to rope_values.bin (with pre+post data)')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots')
    parser.add_argument('--freq-base', type=float, default=500000.0, help='RoPE freq_base (default: 500000 for Llama3)')
    parser.add_argument('--max-tokens', type=int, default=10000, help='Max tokens to load (default: 10000)')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 16, 31], help='Layers to analyze')
    parser.add_argument('--heads', type=int, nargs='+', default=[0], help='Heads to analyze')
    parser.add_argument('--pairs', type=int, nargs='+', default=[0, 8, 16, 32, 48, 63], help='Pair indices to analyze (0-63)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.rope_file), 'rotation_analysis')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("RoPE Rotation Angle Analysis")
    print("=" * 60)

    # Plot 1: Rotation angle (no data needed)
    print("\n[1/5] Rotation angle vs pair index...")
    plot1_rotation_angle_vs_pair(128, args.freq_base, args.output_dir)

    # Load data
    print(f"\nLoading data (max {args.max_tokens} tokens)...")
    data = read_rope_values(args.rope_file, max_tokens=args.max_tokens)

    # Check post-RoPE availability
    has_post = data['layers'][0]['post'] is not None
    if not has_post:
        print("\n⚠ No post-RoPE data found. Computing from pre-RoPE using rotation formula...")
        freqs = compute_rope_freqs(data['n_dims'], args.freq_base)
        for layer_idx in range(data['n_layers']):
            pre = data['layers'][layer_idx]['pre']
            if pre is None:
                continue
            n_tok = pre.shape[0]
            post = np.zeros_like(pre)
            for pair_i in range(data['n_dims'] // 2):
                d0, d1 = pair_i * 2, pair_i * 2 + 1
                for t in range(n_tok):
                    theta = t * freqs[pair_i]
                    cos_t, sin_t = np.cos(theta), np.sin(theta)
                    post[t, :, d0] = pre[t, :, d0] * cos_t - pre[t, :, d1] * sin_t
                    post[t, :, d1] = pre[t, :, d0] * sin_t + pre[t, :, d1] * cos_t
            data['layers'][layer_idx]['post'] = post
        print("  Computed post-RoPE values.")

    # Plot 2-5
    print("\n[2/5] Pair scatter plots...")
    plot2_pair_scatter(data, args.layers, args.heads, args.pairs, args.output_dir)

    print("\n[3/5] Imbalance histograms...")
    plot3_imbalance_histogram(data, args.layers, args.heads, args.pairs, args.output_dir)

    print("\n[4/5] Token-wise imbalance...")
    plot4_tokenwise_imbalance(data, args.layers, args.heads, args.pairs, args.output_dir)

    print("\n[5/5] Rotation direction analysis...")
    plot5_rotation_direction(data, args.layers, args.heads, args.pairs, args.freq_base, args.output_dir)

    print(f"\nDone! All plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
