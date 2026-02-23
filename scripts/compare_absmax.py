#!/usr/bin/env python3
"""
Compare AbsMax (APOR format) between Original and CRS-fused models.
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

MAGIC_APOR = 0x524F5041  # "APOR"
MAGIC_ROPV = 0x524F5056  # "ROPV"

def load_absmax(file_path, target_layer=None):
    """Load AbsMax from .bin file (APOR or ROPV format)"""
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_APOR and magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected {hex(MAGIC_APOR)} or {hex(MAGIC_ROPV)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        
        print(f"Header: layers={n_layers}, heads={n_heads}, dims={n_dims}")
        
        pre_absmax = []
        post_absmax = []
        
        for layer in range(n_layers):
            layer_pre = []
            layer_post = []
            for head in range(n_heads):
                # Read pre
                data_pre = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                layer_pre.append(data_pre)
                # Read post
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

def load_crs_scales(scales_file):
    """Load CRS scales to identify outlier dimensions"""
    MAGIC_SCRS = 0x53435253
    
    with open(scales_file, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_SCRS:
            # Fallback for raw float scales or other formats?
            # Assuming standard SCRS format for now
            print(f"Warning: Invalid magic {hex(magic)}, trying legacy read...")
            f.seek(0)
            # Simple fallback: assume flat float array if file is small?
            # No, user provided 'scales_sqrt_cap_k2.bin', likely SCRS.
            raise ValueError(f"Invalid magic: {hex(magic)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        top_k = struct.unpack('I', f.read(4))[0]

        # Check format based on file size
        header_bytes = 4 * 6
        cur_pos = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(cur_pos, 0)

        remaining = file_size - header_bytes
        per_head_bytes_crs = (top_k * 4) + (top_k * 4) # indices + scales
        per_head_bytes_prs = ((top_k * 2) * 4) + ((top_k * 2) * 4) # indices + scales (pairs)
        
        expected_crs = n_layers * n_heads * per_head_bytes_crs
        expected_prs = n_layers * n_heads * per_head_bytes_prs

        if remaining == expected_prs:
            stored_len = top_k * 2
            print("Detected PRS scale format")
        else:
            stored_len = top_k
            print("Detected CRS scale format")
        
        scales_data = []
        for layer_idx in range(n_layers):
            layer_scales = []
            for head_idx in range(n_heads):
                indices = np.frombuffer(f.read(stored_len * 4), dtype=np.int32)
                scales = np.frombuffer(f.read(stored_len * 4), dtype=np.float32)

                valid_mask = indices >= 0
                layer_scales.append({
                    'indices': indices[valid_mask],
                    'scales': scales[valid_mask]
                })
            scales_data.append(layer_scales)
        
        return scales_data

def visualize_comparison(orig_absmax, crs_absmax, outlier_indices, outlier_scales,
                        layer_idx, head_idx, use_post_rope, output_path=None):

    n_dims = len(orig_absmax)
    dims = np.arange(n_dims)

    rope_stage = "Post-RoPE" if use_post_rope else "Pre-RoPE"

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    base_width = 0.65
    highlight_width = 0.85

    # =========================
    # LEFT: ORIGINAL
    # =========================
    ax1 = axes[0]
    ax1.bar(dims, orig_absmax, color='steelblue',
            alpha=0.6, width=base_width, edgecolor='none')

    used_positions = {}

    valid_indices = [i for i in outlier_indices if i < n_dims]

    if valid_indices:
        ax1.bar(valid_indices, orig_absmax[valid_indices],
                color='red', alpha=0.85,
                width=highlight_width,
                edgecolor='black', linewidth=1.2)

        # stem overlay (깔끔 강조)
        markerline, stemlines, baseline = ax1.stem(
            valid_indices,
            orig_absmax[valid_indices],
            linefmt='r-',
            markerfmt='ro',
            basefmt=" "
        )
        plt.setp(stemlines, linewidth=1.5, alpha=0.6)

        for idx, scale in zip(outlier_indices, outlier_scales):
            if idx >= n_dims:
                continue

            y = orig_absmax[idx]

            # 지그재그 오프셋
            x_offset = -0.25 if idx % 2 == 0 else 0.25

            # 같은 위치면 위로 누적
            level = used_positions.get(idx, 0)
            y_shift = level * 0.8
            used_positions[idx] = level + 1

            ax1.text(
                idx + x_offset,
                y + 0.4 + y_shift,
                f'{idx}\n{y:.2f}',
                fontsize=7,
                ha='center',
                weight='bold',
                color='darkred',
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor='lightyellow',
                          alpha=0.8)
            )

    # Block 0 (0-32)
    ax1.axvspan(0, 32, alpha=0.1, color='red', label='Block 0 (0-32)')
    # Block 1 (32-64)
    ax1.axvspan(32, 64, alpha=0.1, color='orange', label='Block 1 (32-64)')
    # Block 2 (64-96)
    ax1.axvspan(64, 96, alpha=0.1, color='green', label='Block 2 (64-96)')
    # Block 3 (96-128)
    ax1.axvspan(96, 128, alpha=0.1, color='blue', label='Block 3 (96-128)')

    ax1.set_title(f'Original Model\nLayer {layer_idx}, Head {head_idx}',
                  fontsize=12, weight='bold')
    ax1.set_xlim([-2, n_dims + 2])
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.legend()

    # =========================
    # RIGHT: CRS FUSED
    # =========================
    ax2 = axes[1]
    ax2.bar(dims, crs_absmax, color='forestgreen',
            alpha=0.6, width=base_width, edgecolor='none')

    used_positions = {}

    if valid_indices:
        ax2.bar(valid_indices, crs_absmax[valid_indices],
                color='orange', alpha=0.85,
                width=highlight_width,
                edgecolor='black', linewidth=1.2)

        markerline, stemlines, baseline = ax2.stem(
            valid_indices,
            crs_absmax[valid_indices],
            linefmt='orange',
            markerfmt='o',
            basefmt=" "
        )
        plt.setp(stemlines, linewidth=1.5, alpha=0.6)

        for idx, scale in zip(outlier_indices, outlier_scales):
            if idx >= n_dims:
                continue

            y_orig = orig_absmax[idx]
            y_new = crs_absmax[idx]

            reduction = (y_orig - y_new) / y_orig * 100 if y_orig > 1e-6 else 0

            x_offset = -0.25 if idx % 2 == 0 else 0.25
            level = used_positions.get(idx, 0)
            y_shift = level * 0.8
            used_positions[idx] = level + 1

            if reduction >= 0:
                label = f'{idx}\n{y_new:.2f}\n(↓{reduction:.0f}%)'
                facecolor = 'lightgreen'
                textcolor = 'darkgreen'
            else:
                label = f'{idx}\n{y_new:.2f}\n(↑{-reduction:.0f}%)'
                facecolor = 'lightgreen'
                textcolor = 'darkgreen'

            ax2.text(
                idx + x_offset,
                y_new + 0.4 + y_shift,
                label,
                fontsize=6.5,
                ha='center',
                weight='bold',
                color=textcolor,
                bbox=dict(boxstyle='round,pad=0.08',
                          facecolor=facecolor,
                          alpha=0.8)
            )

            # 얇은 화살표
            ax2.annotate(
                '',
                xy=(idx, y_new),
                xytext=(idx, y_orig),
                arrowprops=dict(arrowstyle='->',
                                color='lightcoral',
                                lw=1.5,
                                alpha=0.6)
            )

    # Block 0 (0-32)
    ax2.axvspan(0, 32, alpha=0.1, color='red', label='Block 0 (0-32)')
    # Block 1 (32-64)
    ax2.axvspan(32, 64, alpha=0.1, color='orange', label='Block 1 (32-64)')
    # Block 2 (64-96)
    ax2.axvspan(64, 96, alpha=0.1, color='green', label='Block 2 (64-96)')
    # Block 3 (96-128)
    ax2.axvspan(96, 128, alpha=0.1, color='blue', label='Block 3 (96-128)')

    ax2.set_title(f'PRS-Fused Model\nLayer {layer_idx}, Head {head_idx}',
                  fontsize=12, weight='bold')
    ax2.set_xlim([-2, n_dims + 2])
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend()

    # =========================
    # Y limit sync
    # =========================
    y_max = max(np.max(orig_absmax), np.max(crs_absmax)) * 1.15
    ax1.set_ylim([0, y_max])
    ax2.set_ylim([0, y_max])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved comparison plot to {output_path}")
    else:
        plt.show()

    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"Key Activation Comparison: Layer {layer_idx}, Head {head_idx} ({rope_stage})")
    print(f"{'='*70}")
    print(f"{'Dim':<6} {'Original':<12} {'CRS-Fused':<12} {'Reduction':<12} {'Scale':<10}")
    print("-" * 70)
    
    for idx, scale in zip(outlier_indices, outlier_scales):
        if idx >= n_dims: continue
        orig_val = orig_absmax[idx]
        crs_val = crs_absmax[idx]
        reduction = (orig_val - crs_val) / orig_val * 100 if orig_val > 1e-6 else 0
        print(f"{idx:<6} {orig_val:<12.4f} {crs_val:<12.4f} {reduction:<11.1f}% {scale:<10.4f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Original absmax (mean): {np.mean(orig_absmax):.4f}")
    print(f"  CRS-fused absmax (mean): {np.mean(crs_absmax):.4f}")

def main():
    parser = argparse.ArgumentParser(
        description='Compare AbsMax (APOR) between Original and CRS-fused models'
    )
    parser.add_argument('orig_absmax_file', help='Original model rope_absmax_full.bin (APOR)')
    parser.add_argument('crs_absmax_file', help='CRS-fused model rope_absmax_full.bin (APOR)')
    parser.add_argument('scales_file', help='CRS scales file')
    parser.add_argument('--layer', type=int, default=0, help='Layer index')
    parser.add_argument('--head', type=int, default=0, help='Head index')
    parser.add_argument('--post-rope', action='store_true', help='Use post-RoPE values')
    parser.add_argument('--output', '-o', help='Output image path')
    
    args = parser.parse_args()
    
    print("="*70)
    print("AbsMax Comparison: Original vs PRS-Fused")
    print("="*70)
    
    # Load data
    print(f"Loading {args.orig_absmax_file}...")
    orig_data = load_absmax(args.orig_absmax_file)
    
    print(f"Loading {args.crs_absmax_file}...")
    crs_data = load_absmax(args.crs_absmax_file)
    
    print(f"Loading {args.scales_file}...")
    scales_data = load_crs_scales(args.scales_file)
    
    # Get values
    layer_idx = args.layer
    head_idx = args.head
    
    if args.post_rope:
        orig_vals = orig_data['post_absmax'][layer_idx][head_idx]
        crs_vals = crs_data['post_absmax'][layer_idx][head_idx]
    else:
        orig_vals = orig_data['pre_absmax'][layer_idx][head_idx]
        crs_vals = crs_data['pre_absmax'][layer_idx][head_idx]
        
    # Get outlier info
    head_scales = scales_data[layer_idx][head_idx]
    outlier_indices = head_scales['indices']
    outlier_scales = head_scales['scales']
    
    # Visualize
    visualize_comparison(
        orig_vals, crs_vals, outlier_indices, outlier_scales,
        layer_idx, head_idx, args.post_rope, args.output
    )
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
