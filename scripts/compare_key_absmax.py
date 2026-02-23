#!/usr/bin/env python3
"""
Compare Key absmax between Original and CRS-fused models.
Uses rope_absmax_full.bin files (memory efficient).
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path


def load_absmax_values(absmax_path):
    """Load absmax values from rope_absmax_full.bin."""
    MAGIC_APOR = 0x524F5041  # "APOR"
    
    with open(absmax_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_APOR:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected {hex(MAGIC_APOR)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        
        print(f"Header: layers={n_layers}, heads={n_heads}, dims={n_dims}")
        
        pre_absmax = []
        post_absmax = []
        
        for layer_idx in range(n_layers):
            layer_pre = []
            layer_post = []
            for head_idx in range(n_heads):
                pre = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                post = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                layer_pre.append(pre)
                layer_post.append(post)
            pre_absmax.append(layer_pre)
            post_absmax.append(layer_post)
    
    return {
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_dims': n_dims,
        'pre_absmax': pre_absmax,   # [layer][head][dim]
        'post_absmax': post_absmax  # [layer][head][dim]
    }


def load_crs_scales(scales_path):
    """Load CRS scales from binary file."""
    MAGIC_SCRS = 0x53435253
    
    with open(scales_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_SCRS:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        k = struct.unpack('I', f.read(4))[0]

        # PRS format: header k denotes number of pairs, but each head stores 2*k dims.
        # Legacy CRS format: header k equals number of dims stored.
        header_bytes = 4 * 6
        cur_pos = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(cur_pos, 0)

        remaining = file_size - header_bytes
        per_head_bytes_crs = (k * 4) + (k * 4)
        per_head_bytes_prs = ((k * 2) * 4) + ((k * 2) * 4)
        expected_crs = n_layers * n_heads * per_head_bytes_crs
        expected_prs = n_layers * n_heads * per_head_bytes_prs

        if remaining == expected_prs:
            stored_len = k * 2
        elif remaining == expected_crs:
            stored_len = k
        else:
            stored_len = k * 2
        
        scales = []
        for _ in range(n_layers):
            layer_scales = []
            for _ in range(n_heads):
                indices = np.frombuffer(f.read(stored_len * 4), dtype=np.int32)
                scale_values = np.frombuffer(f.read(stored_len * 4), dtype=np.float32)
                layer_scales.append({
                    'indices': indices,
                    'scales': scale_values
                })
            scales.append(layer_scales)
    
    return {
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_dims': n_dims,
        'k': k,
        'scales': scales
    }


def visualize_comparison(orig_absmax, crs_absmax, indices, scales, n_dims,
                        layer_idx, head_idx, use_post_rope=False, output_path=None):
    """Create side-by-side comparison visualization."""
    
    dims = np.arange(n_dims)
    rope_stage = "Post-RoPE" if use_post_rope else "Pre-RoPE"
    
    # Filter valid outlier indices
    valid_mask = indices >= 0
    outlier_indices = indices[valid_mask]
    outlier_scales = scales[valid_mask]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Left: Original model
    ax1 = axes[0]
    ax1.bar(dims, orig_absmax, color='steelblue', alpha=0.7, width=1.0, edgecolor='none')
    
    # Highlight outlier dimensions
    if len(outlier_indices) > 0:
        ax1.bar(outlier_indices, orig_absmax[outlier_indices], 
               color='red', alpha=0.9, width=1.2, edgecolor='black', linewidth=2,
               label=f'Outliers (n={len(outlier_indices)})')
        
        # Add dimension labels
        for idx in outlier_indices:
            y_pos = orig_absmax[idx]
            ax1.text(idx, y_pos + 0.5, f'{idx}', 
                    fontsize=9, color='red', ha='center', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    ax1.axvspan(80, 128, alpha=0.15, color='green', label='Target Range (80-128)')
    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('Absmax Value', fontsize=12)
    ax1.set_title(f'Original Model\nLayer {layer_idx}, Head {head_idx} ({rope_stage})', 
                 fontsize=13, weight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim([-2, n_dims + 2])
    
    # Right: CRS-fused model
    ax2 = axes[1]
    ax2.bar(dims, crs_absmax, color='forestgreen', alpha=0.7, width=1.0, edgecolor='none')
    
    # Highlight outlier dimensions (should be suppressed)
    if len(outlier_indices) > 0:
        ax2.bar(outlier_indices, crs_absmax[outlier_indices], 
               color='orange', alpha=0.9, width=1.2, edgecolor='black', linewidth=2,
               label=f'Suppressed Outliers (n={len(outlier_indices)})')
        
        # Show reduction
        for idx, scale in zip(outlier_indices, outlier_scales):
            y_orig = orig_absmax[idx]
            y_crs = crs_absmax[idx]
            reduction = (y_orig - y_crs) / y_orig * 100 if y_orig > 0 else 0
            
            ax2.text(idx, y_crs + 0.5, f'{idx}\n↓{reduction:.0f}%', 
                    fontsize=9, color='darkgreen', ha='center', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            
            # Draw arrow showing reduction
            if y_orig > y_crs:
                ax2.annotate('', xy=(idx, y_crs), xytext=(idx, y_orig),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.6))
    
    ax2.axvspan(80, 128, alpha=0.15, color='green', label='Target Range (80-128)')
    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('Absmax Value', fontsize=12)
    ax2.set_title(f'CRS-Fused Model\nLayer {layer_idx}, Head {head_idx} ({rope_stage})', 
                 fontsize=13, weight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim([-2, n_dims + 2])
    
    # Match y-axis limits
    y_max = max(np.max(orig_absmax), np.max(crs_absmax)) * 1.15
    ax1.set_ylim([0, y_max])
    ax2.set_ylim([0, y_max])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved comparison plot to {output_path}")
    else:
        plt.show()
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"Key Absmax Comparison: Layer {layer_idx}, Head {head_idx} ({rope_stage})")
    print(f"{'='*70}")
    print(f"{'Dim':<6} {'Original':<12} {'CRS-Fused':<12} {'Reduction':<12} {'Scale':<10}")
    print("-" * 70)
    
    for idx, scale in zip(outlier_indices, outlier_scales):
        orig_val = orig_absmax[idx]
        crs_val = crs_absmax[idx]
        reduction = (orig_val - crs_val) / orig_val * 100 if orig_val > 0 else 0
        print(f"{idx:<6} {orig_val:<12.4f} {crs_val:<12.4f} {reduction:<11.1f}% {scale:<10.4f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Original absmax (mean): {np.mean(orig_absmax):.4f}")
    print(f"  CRS-fused absmax (mean): {np.mean(crs_absmax):.4f}")
    print(f"  Outlier dimensions: {len(outlier_indices)}")
    
    if len(outlier_indices) > 0:
        orig_outlier_mean = np.mean(orig_absmax[outlier_indices])
        crs_outlier_mean = np.mean(crs_absmax[outlier_indices])
        overall_reduction = (orig_outlier_mean - crs_outlier_mean) / orig_outlier_mean * 100
        print(f"  Outlier mean reduction: {overall_reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Compare Key absmax between Original and CRS-fused models'
    )
    parser.add_argument('orig_absmax_file', help='Original model rope_absmax_full.bin')
    parser.add_argument('crs_absmax_file', help='CRS-fused model rope_absmax_full.bin')
    parser.add_argument('scales_file', help='CRS scales file')
    parser.add_argument('--layer', type=int, default=0, help='Layer index (default: 0)')
    parser.add_argument('--head', type=int, default=0, help='Head index (default: 0)')
    parser.add_argument('--post-rope', action='store_true', help='Use post-RoPE values')
    parser.add_argument('--output', '-o', help='Output image path')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Key Absmax Comparison: Original vs CRS-Fused")
    print("="*70)
    
    print(f"\n[1/3] Loading original model absmax...")
    orig_data = load_absmax_values(args.orig_absmax_file)
    
    print(f"\n[2/3] Loading CRS-fused model absmax...")
    crs_data = load_absmax_values(args.crs_absmax_file)
    
    print(f"\n[3/3] Loading CRS scales...")
    scales_data = load_crs_scales(args.scales_file)
    
    # Get absmax for specified layer and head
    layer_idx = args.layer
    head_idx = args.head
    
    if args.post_rope:
        orig_absmax = orig_data['post_absmax'][layer_idx][head_idx]
        crs_absmax = crs_data['post_absmax'][layer_idx][head_idx]
        print(f"\nUsing post-RoPE absmax")
    else:
        orig_absmax = orig_data['pre_absmax'][layer_idx][head_idx]
        crs_absmax = crs_data['pre_absmax'][layer_idx][head_idx]
        print(f"\nUsing pre-RoPE absmax")
    
    # Get CRS scales for this layer and head
    head_scales = scales_data['scales'][layer_idx][head_idx]
    indices = head_scales['indices']
    scales = head_scales['scales']
    
    n_dims = orig_data['n_dims']
    
    # Visualize
    visualize_comparison(
        orig_absmax, crs_absmax, indices, scales, n_dims,
        layer_idx, head_idx, args.post_rope, args.output
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
