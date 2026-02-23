#!/usr/bin/env python3
"""
Visualize CRS effect on Key absmax values from calibration data.
Uses rope_absmax_full.bin which is memory-efficient.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_absmax_values(absmax_path):
    """Load absmax values from rope_absmax_full.bin."""
    import struct
    
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
    with open(scales_path, 'rb') as f:
        n_layers = np.frombuffer(f.read(4), dtype=np.int32)[0]
        n_heads = np.frombuffer(f.read(4), dtype=np.int32)[0]
        n_dims = np.frombuffer(f.read(4), dtype=np.int32)[0]
        k = np.frombuffer(f.read(4), dtype=np.int32)[0]
        
        scales = []
        for _ in range(n_layers):
            layer_scales = []
            for _ in range(n_heads):
                indices = np.frombuffer(f.read(k * 4), dtype=np.int32)
                scale_values = np.frombuffer(f.read(k * 4), dtype=np.float32)
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


def apply_crs_to_absmax(absmax, indices, scales, n_dims):
    """Apply CRS scaling to absmax values."""
    scaled = absmax.copy()
    
    for idx, scale in zip(indices, scales):
        if idx < n_dims:
            scaled[idx] = absmax[idx] / scale
    
    return scaled


def visualize_absmax_comparison(original_absmax, crs_absmax, layer_idx, head_idx,
                                 indices, scales, n_dims, use_post_rope=False,
                                 output_path=None):
    """Create visualization comparing original vs CRS-scaled absmax."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    dims = np.arange(n_dims)
    
    # Mark modified dimensions
    modified_dims = set(indices)
    modified_mask = np.array([d in modified_dims for d in dims])
    
    rope_stage = "Post-RoPE" if use_post_rope else "Pre-RoPE"
    
    # Plot 1: Absmax comparison
    ax1 = axes[0]
    ax1.plot(dims, original_absmax, 'o-', label='Original K', alpha=0.7, markersize=4, color='blue')
    ax1.plot(dims, crs_absmax, 's-', label='CRS Scaled K', alpha=0.7, markersize=4, color='green')
    
    # Highlight modified dimensions
    if len(indices) > 0:
        ax1.scatter(indices, original_absmax[indices], c='red', s=150, marker='o', 
                   edgecolors='black', linewidths=2, label='Modified (Original)', zorder=5)
        ax1.scatter(indices, crs_absmax[indices], c='orange', s=150, marker='s',
                   edgecolors='black', linewidths=2, label='Modified (CRS)', zorder=5)
        
        # Add scale annotations
        for idx, scale in zip(indices, scales):
            y_pos = crs_absmax[idx]
            ax1.annotate(f'÷{scale:.2f}', 
                        xy=(idx, y_pos),
                        xytext=(0, -20), textcoords='offset points',
                        ha='center', fontsize=9, color='red', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
            
            # Draw arrow showing reduction
            ax1.annotate('', xy=(idx, y_pos), xytext=(idx, original_absmax[idx]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.6))
    
    ax1.axvspan(80, 128, alpha=0.15, color='green', label='Target Range (80-128)')
    ax1.set_xlabel('Dimension', fontsize=11)
    ax1.set_ylabel('Absmax Value', fontsize=11)
    ax1.set_title(f'Layer {layer_idx}, Head {head_idx} ({rope_stage}): Key Absmax per Dimension\nCRS Effect on Outlier Suppression', 
                  fontsize=12, weight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reduction Ratio
    ax2 = axes[1]
    ratio = np.where(original_absmax > 1e-10, crs_absmax / original_absmax, 1.0)
    
    # Bar plot for better visibility
    colors = ['red' if m else 'lightblue' for m in modified_mask]
    ax2.bar(dims, ratio, color=colors, alpha=0.7, width=1.0, edgecolor='none')
    
    if len(indices) > 0:
        expected_ratios = 1.0 / scales
        
        # Highlight modified dimensions
        ax2.bar(indices, ratio[indices], color='orange', alpha=0.9, width=1.2,
               edgecolor='black', linewidth=2, label='Modified Dimensions')
        
        # Show expected ratio as horizontal lines
        for idx, expected_ratio in zip(indices, expected_ratios):
            ax2.plot([idx-0.5, idx+0.5], [expected_ratio, expected_ratio], 
                    color='green', linestyle='--', linewidth=3, alpha=0.8)
            ax2.text(idx, expected_ratio + 0.05, f'{expected_ratio:.3f}',
                    fontsize=8, color='green', ha='center', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=2, alpha=0.7, label='No change (ratio=1.0)')
    ax2.axvspan(80, 128, alpha=0.15, color='green')
    ax2.set_xlabel('Dimension', fontsize=11)
    ax2.set_ylabel('Ratio (CRS / Original)', fontsize=11)
    ax2.set_title(f'Outlier Reduction Ratio: Modified Dimensions Scaled Down by ~1/3', 
                  fontsize=12, weight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.5])
    ax2.set_xlim([-2, n_dims + 2])
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize CRS effect on Key absmax values from calibration data'
    )
    parser.add_argument('absmax_file', help='Path to rope_absmax_full.bin')
    parser.add_argument('scales_file', help='Path to CRS scales binary file')
    parser.add_argument('--layer', type=int, default=18, help='Layer index (default: 18)')
    parser.add_argument('--head', type=int, default=5, help='Head index (default: 5)')
    parser.add_argument('--post-rope', action='store_true', help='Use post-RoPE values instead of pre-RoPE')
    parser.add_argument('--output', '-o', help='Output image path (default: show plot)')
    
    args = parser.parse_args()
    
    print(f"Loading absmax values from {args.absmax_file}...")
    absmax_data = load_absmax_values(args.absmax_file)
    
    print(f"Loading CRS scales from {args.scales_file}...")
    scales_data = load_crs_scales(args.scales_file)
    
    n_dims = absmax_data['n_dims']
    
    # Get absmax for specified layer and head
    if args.post_rope:
        original_absmax = absmax_data['post_absmax'][args.layer][args.head]
        print(f"Using post-RoPE absmax")
    else:
        original_absmax = absmax_data['pre_absmax'][args.layer][args.head]
        print(f"Using pre-RoPE absmax")
    
    # Get CRS scales for this layer and head
    head_scales = scales_data['scales'][args.layer][args.head]
    indices = head_scales['indices']
    scales = head_scales['scales']
    
    print(f"\nLayer {args.layer}, Head {args.head} CRS Info:")
    print(f"  Modified dimensions: {len(indices)}")
    if len(indices) > 0:
        print(f"  Dimensions: {indices}")
        print(f"  Scales: {scales}")
    else:
        print(f"  No modifications (no outliers detected)")
        return 0
    
    # Apply CRS scaling
    print(f"\nApplying CRS scaling to absmax...")
    crs_absmax = apply_crs_to_absmax(original_absmax, indices, scales, n_dims)
    
    # Statistics
    print(f"\nAbsmax Statistics (CRS Effect):")
    print(f"{'Dim':<6} {'Original':<12} {'CRS':<12} {'Ratio':<12} {'Expected':<12} {'Status'}")
    print("-" * 70)
    for idx, scale in zip(indices, scales):
        orig_val = original_absmax[idx]
        crs_val = crs_absmax[idx]
        ratio = crs_val / orig_val if orig_val > 1e-10 else 1.0
        expected = 1.0 / scale
        status = "✓ PASS" if abs(ratio - expected) < 0.01 else "✗ FAIL"
        print(f"{idx:<6} {orig_val:<12.6f} {crs_val:<12.6f} {ratio:<12.6f} {expected:<12.6f} {status}")
    
    print(f"\nGenerating visualization...")
    visualize_absmax_comparison(
        original_absmax, crs_absmax, args.layer, args.head,
        indices, scales, n_dims, args.post_rope, args.output
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
