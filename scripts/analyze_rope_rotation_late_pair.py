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
import csv

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


def load_rotation_csv(csv_path):
    """Return {(layer, head, pair): alpha_rad} from CSV."""
    rotations = {}
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['layer']), int(row['head']), int(row['pair']))
            rotations[key] = float(row['suggested_alpha_rad'])
    return rotations


def rotate_points(x, y, alpha):
    c = np.cos(alpha)
    s = np.sin(alpha)
    return c * x - s * y, s * x + c * y


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


def plot2_pair_scatter(data, layers, heads, pairs, output_dir, rotations=None):
    """Plot 2: Scatter of (x,y) pairs pre vs post RoPE, with optional rotated-post simulation."""
    n_pairs_to_show = len(pairs)
    n_layers_to_show = len(layers)
    n_panels_per_pair = 3 if rotations is not None else 2

    for head_idx in heads:
        fig, axes = plt.subplots(
            n_layers_to_show,
            n_pairs_to_show * n_panels_per_pair,
            figsize=(5 * n_pairs_to_show * n_panels_per_pair, 5 * n_layers_to_show),
        )
        if n_layers_to_show == 1:
            axes = axes[np.newaxis, :]

        for li, layer_idx in enumerate(layers):
            pre = data['layers'][layer_idx]['pre']
            post = data['layers'][layer_idx]['post']
            if pre is None or post is None:
                continue

            n_tokens_use = min(pre.shape[0], post.shape[0])

            for pi, pair_idx in enumerate(pairs):
                d0 = pair_idx * 2
                d1 = pair_idx * 2 + 1
                col_base = pi * n_panels_per_pair

                # Pre-RoPE scatter
                x_pre = pre[:n_tokens_use, head_idx, d0]
                y_pre = pre[:n_tokens_use, head_idx, d1]
                ax = axes[li, col_base]
                ax.scatter(x_pre, y_pre, alpha=0.1, s=1, c='blue')
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
                ax = axes[li, col_base + 1]
                ax.scatter(x_post, y_post, alpha=0.1, s=1, c='red')
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

                if rotations is not None:
                    alpha = rotations.get((layer_idx, head_idx, pair_idx), 0.0)
                    x_rot, y_rot = rotate_points(x_post, y_post, alpha)
                    ax = axes[li, col_base + 2]
                    ax.scatter(x_rot, y_rot, alpha=0.1, s=1, c='purple')
                    ax.set_title(
                        f'Rotated-Post L{layer_idx} H{head_idx} P{pair_idx}\nα={np.degrees(alpha):.2f} deg',
                        fontsize=9,
                    )
                    ax.set_xlabel(f'dim {d0}')
                    ax.set_ylabel(f'dim {d1}')
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                    lim_rot = max(np.abs(x_rot).max(), np.abs(y_rot).max()) * 1.1
                    ax.set_xlim(-lim_rot, lim_rot)
                    ax.set_ylim(-lim_rot, lim_rot)
                    ax.axhline(0, color='gray', lw=0.5)
                    ax.axvline(0, color='gray', lw=0.5)

        if rotations is None:
            title = f'Pair Scatter: Pre-RoPE (blue) vs Post-RoPE (red) — Head {head_idx}'
        else:
            title = f'Pair Scatter: Pre (blue), Post (red), Rotated-Post Sim (purple) — Head {head_idx}'
        plt.suptitle(title, fontsize=14)
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

def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def axis_angle_diff(a, b):
    """
    Principal-axis angle difference in [0, pi/2].
    PCA axis is sign-ambiguous, so compare modulo pi.
    """
    return 0.5 * np.abs(np.arctan2(np.sin(2 * (a - b)), np.cos(2 * (a - b))))


def nearest_diag_angle(phi):
    """
    Return nearest diagonal direction among:
    pi/4, -pi/4, 3pi/4, -3pi/4
    """
    diags = np.array([np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4], dtype=np.float64)
    diffs = np.abs(np.arctan2(np.sin(diags - phi), np.cos(diags - phi)))
    return diags[np.argmin(diffs)]


def compute_pair_shape_stats(x, y, eps=1e-12):
    """
    Compute shape stats for a 2D point cloud z_t = (x_t, y_t).
    """
    z = np.stack([x, y], axis=1).astype(np.float64)
    mu = z.mean(axis=0)                              # mean vector
    centered = z - mu
    cov = (centered.T @ centered) / max(len(z), 1)  # 2x2 covariance

    eigvals, eigvecs = np.linalg.eigh(cov)          # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    lam1 = float(max(eigvals[0], 0.0))
    lam2 = float(max(eigvals[1], 0.0))
    v1 = eigvecs[:, 0]

    spread = lam1 + lam2
    anisotropy = (lam1 - lam2) / (spread + eps)     # 0 ~ 1
    mean_norm2 = float(mu @ mu)
    mean_compactness = mean_norm2 / (mean_norm2 + spread + eps)  # 0 ~ 1

    # pair magnitude balance: |x| and |y| being similar is good for shared-scale quant
    absx = np.abs(x)
    absy = np.abs(y)
    balance = 1.0 - np.mean(np.abs(absx - absy) / (absx + absy + eps))  # 0 ~ 1, high=balanced

    mu_norm = np.sqrt(mean_norm2)
    mu_angle = float(np.arctan2(mu[1], mu[0])) if mu_norm > 1e-10 else np.nan
    pca_angle = float(np.arctan2(v1[1], v1[0]))

    return {
        'mu_x': float(mu[0]),
        'mu_y': float(mu[1]),
        'mu_norm': float(mu_norm),
        'lam1': lam1,
        'lam2': lam2,
        'spread': float(spread),
        'anisotropy': float(anisotropy),
        'mean_compactness': float(mean_compactness),
        'balance': float(balance),
        'mu_angle': mu_angle,
        'pca_angle': pca_angle,
    }


def compute_pair_candidate_table(data, layers, heads, max_points=5000):
    """
    Build candidate table for all requested layer/head/pair using PRE-RoPE stats,
    plus optional PRE->POST drift stats if post exists.
    """
    rows = []
    n_pairs = data['n_dims'] // 2

    for layer_idx in layers:
        pre = data['layers'][layer_idx]['pre']
        post = data['layers'][layer_idx]['post']
        if pre is None:
            continue

        for head_idx in heads:
            for pair_idx in range(n_pairs):
                d0 = pair_idx * 2
                d1 = pair_idx * 2 + 1

                n_use = min(pre.shape[0], max_points)
                x_pre = pre[:n_use, head_idx, d0]
                y_pre = pre[:n_use, head_idx, d1]

                pre_stats = compute_pair_shape_stats(x_pre, y_pre)

                row = {
                    'layer': layer_idx,
                    'head': head_idx,
                    'pair': pair_idx,
                    'dim0': d0,
                    'dim1': d1,

                    'pre_spread': pre_stats['spread'],
                    'pre_anisotropy': pre_stats['anisotropy'],
                    'pre_mean_compactness': pre_stats['mean_compactness'],
                    'pre_balance': pre_stats['balance'],
                    'pre_mu_norm': pre_stats['mu_norm'],
                    'pre_mu_angle': pre_stats['mu_angle'],
                    'pre_pca_angle': pre_stats['pca_angle'],
                    'pre_lam1': pre_stats['lam1'],
                    'pre_lam2': pre_stats['lam2'],
                }

                # Optional post stats / drift
                if post is not None:
                    n_post = min(post.shape[0], max_points)
                    n_both = min(n_use, n_post)
                    x_post = post[:n_both, head_idx, d0]
                    y_post = post[:n_both, head_idx, d1]
                    post_stats = compute_pair_shape_stats(x_post, y_post)

                    row.update({
                        'post_spread': post_stats['spread'],
                        'post_anisotropy': post_stats['anisotropy'],
                        'post_mean_compactness': post_stats['mean_compactness'],
                        'post_balance': post_stats['balance'],
                        'post_mu_norm': post_stats['mu_norm'],
                        'post_mu_angle': post_stats['mu_angle'],
                        'post_pca_angle': post_stats['pca_angle'],
                        'post_lam1': post_stats['lam1'],
                        'post_lam2': post_stats['lam2'],
                    })

                    if np.isfinite(pre_stats['mu_angle']) and np.isfinite(post_stats['mu_angle']):
                        row['mu_drift'] = float(np.abs(wrap_angle(post_stats['mu_angle'] - pre_stats['mu_angle'])))
                    else:
                        row['mu_drift'] = np.nan

                    row['pca_drift'] = float(axis_angle_diff(post_stats['pca_angle'], pre_stats['pca_angle']))
                else:
                    row.update({
                        'post_spread': np.nan,
                        'post_anisotropy': np.nan,
                        'post_mean_compactness': np.nan,
                        'post_balance': np.nan,
                        'post_mu_norm': np.nan,
                        'post_mu_angle': np.nan,
                        'post_pca_angle': np.nan,
                        'post_lam1': np.nan,
                        'post_lam2': np.nan,
                        'mu_drift': np.nan,
                        'pca_drift': np.nan,
                    })

                rows.append(row)

    # ---- normalize score within each (layer, head) ----
    # spread는 절대값이라 layer/head마다 scale이 달라질 수 있어서 group-wise normalization
    grouped = {}
    for r in rows:
        key = (r['layer'], r['head'])
        grouped.setdefault(key, []).append(r)

    for key, group in grouped.items():
        spreads = np.array([g['pre_spread'] for g in group], dtype=np.float64)
        med_spread = float(np.median(spreads)) + 1e-12

        for g in group:
            spread_score = np.exp(-g['pre_spread'] / med_spread)

            # mean-dominant vs pca-dominant
            if g['pre_mean_compactness'] >= g['pre_anisotropy']:
                mode = 'mean'
                base_angle = g['pre_mu_angle']
                drift = g['mu_drift'] if np.isfinite(g['mu_drift']) else 0.0
            else:
                mode = 'pca'
                base_angle = g['pre_pca_angle']
                drift = g['pca_drift'] if np.isfinite(g['pca_drift']) else 0.0

            drift_score = np.exp(-drift / 0.35)  # smaller drift -> better
            orientation_score = max(g['pre_mean_compactness'], g['pre_anisotropy'])
            balance_score = g['pre_balance']

            # final score: compact + directional + balanced + stable
            score = spread_score * orientation_score * balance_score * drift_score

            if np.isfinite(base_angle):
                target_diag = nearest_diag_angle(base_angle)
                alpha = wrap_angle(target_diag - base_angle)
            else:
                target_diag = np.nan
                alpha = np.nan

            g['spread_score'] = float(spread_score)
            g['orientation_score'] = float(orientation_score)
            g['drift_score'] = float(drift_score)
            g['candidate_mode'] = mode
            g['target_diag_angle'] = float(target_diag) if np.isfinite(target_diag) else np.nan
            g['suggested_alpha_rad'] = float(alpha) if np.isfinite(alpha) else np.nan
            g['suggested_alpha_deg'] = float(np.degrees(alpha)) if np.isfinite(alpha) else np.nan
            g['candidate_score'] = float(score)

            # hard label for quick filtering
            # 너무 퍼져 있거나 방향성이 약하면 skip
            if (g['spread_score'] < 0.35) or (orientation_score < 0.25):
                g['candidate_label'] = 'skip'
            elif mode == 'mean':
                g['candidate_label'] = 'mean_cluster'
            else:
                g['candidate_label'] = 'elongated_cluster'

    rows.sort(key=lambda x: (x['layer'], x['head'], -x['candidate_score']))
    return rows


def save_pair_candidate_csv(rows, output_dir, topk_per_group=12):
    csv_path = os.path.join(output_dir, 'pair_candidates.csv')
    if len(rows) == 0:
        print("  [warn] No candidate rows to save.")
        return

    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved CSV: {csv_path}")

    # Also save a smaller top-k file
    grouped = {}
    for r in rows:
        key = (r['layer'], r['head'])
        grouped.setdefault(key, []).append(r)

    top_rows = []
    for key, group in grouped.items():
        top_rows.extend(sorted(group, key=lambda x: x['candidate_score'], reverse=True)[:topk_per_group])

    top_csv_path = os.path.join(output_dir, f'pair_candidates_top{topk_per_group}.csv')
    with open(top_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_rows)

    print(f"  Saved CSV: {top_csv_path}")


def plot6_candidate_score_by_pair(rows, layers, heads, output_dir):
    """
    Simple line plot: candidate score vs pair index for each layer/head.
    """
    for head_idx in heads:
        fig, axes = plt.subplots(len(layers), 1, figsize=(12, 3.5 * len(layers)), sharex=True)
        if len(layers) == 1:
            axes = [axes]

        for ai, layer_idx in enumerate(layers):
            group = [r for r in rows if r['layer'] == layer_idx and r['head'] == head_idx]
            if not group:
                continue
            group = sorted(group, key=lambda x: x['pair'])

            xs = [r['pair'] for r in group]
            ys = [r['candidate_score'] for r in group]
            colors = []
            for r in group:
                if r['candidate_label'] == 'skip':
                    colors.append('gray')
                elif r['candidate_label'] == 'mean_cluster':
                    colors.append('blue')
                else:
                    colors.append('red')

            axes[ai].scatter(xs, ys, c=colors, s=18, alpha=0.9)
            axes[ai].plot(xs, ys, alpha=0.35, lw=0.8)
            axes[ai].set_title(f'Candidate score vs pair index — L{layer_idx} H{head_idx}')
            axes[ai].set_ylabel('score')
            axes[ai].grid(True, alpha=0.3)

        axes[-1].set_xlabel('pair index')
        plt.tight_layout()
        path = os.path.join(output_dir, f'plot6_candidate_score_h{head_idx}.png')
        plt.savefig(path, dpi=130)
        plt.close()
        print(f"  Saved: {path}")


def print_top_candidates(rows, topk=10):
    print("\nTop candidate pairs:")
    for r in sorted(rows, key=lambda x: x['candidate_score'], reverse=True)[:topk]:
        print(
            f"  L{r['layer']:02d} H{r['head']:02d} P{r['pair']:02d} "
            f"score={r['candidate_score']:.4f} "
            f"label={r['candidate_label']} "
            f"mode={r['candidate_mode']} "
            f"spread={r['pre_spread']:.4f} "
            f"aniso={r['pre_anisotropy']:.4f} "
            f"mcompact={r['pre_mean_compactness']:.4f} "
            f"balance={r['pre_balance']:.4f} "
            f"alpha={r['suggested_alpha_deg']:.2f} deg"
        )


def main():
    parser = argparse.ArgumentParser(description='RoPE Rotation Angle Analysis')
    parser.add_argument('rope_file', help='Path to rope_values.bin (with pre+post data)')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots')
    parser.add_argument('--freq-base', type=float, default=500000.0, help='RoPE freq_base (default: 500000 for Llama3)')
    parser.add_argument('--max-tokens', type=int, default=10000, help='Max tokens to load (default: 10000)')
    parser.add_argument('--layers', type=int, nargs='+', default=[0, 16, 31], help='Layers to analyze')
    parser.add_argument('--heads', type=int, nargs='+', default=[0], help='Heads to analyze')
    parser.add_argument('--pairs', type=int, nargs='+', default=[0, 8, 16, 32, 48, 63], help='Pair indices to analyze (0-63)')
    parser.add_argument('--rotation-csv', default=None, help='Optional CSV with suggested_alpha_rad for simulated rotated-post scatter')

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
    rotations = None
    if args.rotation_csv is not None:
        rotations = load_rotation_csv(args.rotation_csv)
        print(f"  Loaded rotation CSV: {args.rotation_csv} ({len(rotations)} entries)")
    plot2_pair_scatter(data, args.layers, args.heads, args.pairs, args.output_dir, rotations=rotations)

    print("\n[3/5] Imbalance histograms...")
    plot3_imbalance_histogram(data, args.layers, args.heads, args.pairs, args.output_dir)

    print("\n[4/5] Token-wise imbalance...")
    plot4_tokenwise_imbalance(data, args.layers, args.heads, args.pairs, args.output_dir)

    print("\n[5/5] Rotation direction analysis...")
    plot5_rotation_direction(data, args.layers, args.heads, args.pairs, args.freq_base, args.output_dir)

    print("\n[6/7] Pair candidate scoring...")
    candidate_rows = compute_pair_candidate_table(
        data,
        layers=args.layers,
        heads=args.heads,
        max_points=min(args.max_tokens, 5000),
    )
    save_pair_candidate_csv(candidate_rows, args.output_dir, topk_per_group=12)
    plot6_candidate_score_by_pair(candidate_rows, args.layers, args.heads, args.output_dir)
    print_top_candidates(candidate_rows, topk=20)


    print(f"\nDone! All plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
