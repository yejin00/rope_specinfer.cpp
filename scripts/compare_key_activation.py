#!/usr/bin/env python3
"""
Compare Key activations between Original and CRS-fused models.
Visualize side-by-side with outlier dimensions highlighted.
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path


def load_rope_values(rope_file, target_layer=None):
    """Load Key activations from rope_values_full.bin (only target layer to save memory)"""
    MAGIC_ROPV = 0x524F5056  # "ROPV" in file byte order
    
    with open(rope_file, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected {hex(MAGIC_ROPV)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        total_tokens = struct.unpack('I', f.read(4))[0]
        
        print(f"Header: layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={total_tokens}")
        
        layer_data = None
        
        for layer_idx in range(n_layers):
            pre_count = struct.unpack('I', f.read(4))[0]
            post_count = struct.unpack('I', f.read(4))[0]
            
            pre_size = pre_count * n_heads * n_dims * 4
            post_size = post_count * n_heads * n_dims * 4
            
            if target_layer is not None and layer_idx == target_layer:
                # Load only target layer
                pre_data = np.frombuffer(f.read(pre_size), dtype=np.float32)
                post_data = np.frombuffer(f.read(post_size), dtype=np.float32)
                
                pre_data = pre_data.reshape(n_heads, pre_count, n_dims)
                post_data = post_data.reshape(n_heads, post_count, n_dims)
                
                layer_data = {
                    'pre': pre_data,
                    'post': post_data,
                    'pre_count': pre_count,
                    'post_count': post_count
                }
                print(f"  Loaded Layer {layer_idx}: pre={pre_count} tokens, post={post_count} tokens")
            else:
                # Skip this layer
                f.seek(pre_size + post_size, 1)  # Seek forward
        
        if target_layer is not None and layer_data is None:
            raise ValueError(f"Layer {target_layer} not found in file")
        
        return {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'layer_data': layer_data
        }


def load_crs_scales(scales_file):
    """Load CRS scales to identify outlier dimensions"""
    MAGIC_SCRS = 0x53435253
    
    with open(scales_file, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_SCRS:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        top_k = struct.unpack('I', f.read(4))[0]

        # PRS format: header top_k denotes number of pairs, but each head stores 2*top_k dims.
        # Legacy CRS format: header top_k equals number of dims stored.
        header_bytes = 4 * 6
        cur_pos = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(cur_pos, 0)

        remaining = file_size - header_bytes
        per_head_bytes_crs = (top_k * 4) + (top_k * 4)
        per_head_bytes_prs = ((top_k * 2) * 4) + ((top_k * 2) * 4)
        expected_crs = n_layers * n_heads * per_head_bytes_crs
        expected_prs = n_layers * n_heads * per_head_bytes_prs

        if remaining == expected_prs:
            stored_len = top_k * 2
        elif remaining == expected_crs:
            stored_len = top_k
        else:
            stored_len = top_k * 2
        
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
        
        return {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'scales': scales_data
        }


def compute_absmax(activations):
    """Compute channel-wise absmax across all tokens"""
    # activations: [n_tokens, n_dims]
    return np.max(np.abs(activations), axis=0)


def visualize_comparison(orig_absmax, crs_absmax, outlier_indices, outlier_scales,
                        layer_idx, head_idx, use_post_rope, output_path=None):
    """Create side-by-side comparison visualization"""
    
    n_dims = len(orig_absmax)
    dims = np.arange(n_dims)
    
    rope_stage = "Post-RoPE" if use_post_rope else "Pre-RoPE"
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Left: Original model
    ax1 = axes[0]
    ax1.bar(dims, orig_absmax, color='steelblue', alpha=0.7, width=1.0, edgecolor='none')
    
    # Highlight outlier dimensions
    if len(outlier_indices) > 0:
        ax1.bar(outlier_indices, orig_absmax[outlier_indices], 
               color='red', alpha=0.9, width=1.2, edgecolor='black', linewidth=2)
        
        # Add scale annotations
        for idx, scale in zip(outlier_indices, outlier_scales):
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
               color='orange', alpha=0.9, width=1.2, edgecolor='black', linewidth=2)
        
        # Show reduction
        for idx, scale in zip(outlier_indices, outlier_scales):
            y_orig = orig_absmax[idx]
            y_crs = crs_absmax[idx]
            reduction = (y_orig - y_crs) / y_orig * 100
            
            ax2.text(idx, y_crs + 0.5, f'{idx}\n↓{reduction:.0f}%', 
                    fontsize=9, color='darkgreen', ha='center', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            
            # Draw arrow showing reduction
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
    y_max = max(np.max(orig_absmax), np.max(crs_absmax)) * 1.1
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
    print(f"Key Activation Comparison: Layer {layer_idx}, Head {head_idx} ({rope_stage})")
    print(f"{'='*70}")
    print(f"{'Dim':<6} {'Original':<12} {'CRS-Fused':<12} {'Reduction':<12} {'Scale':<10}")
    print("-" * 70)
    
    for idx, scale in zip(outlier_indices, outlier_scales):
        orig_val = orig_absmax[idx]
        crs_val = crs_absmax[idx]
        reduction = (orig_val - crs_val) / orig_val * 100
        print(f"{idx:<6} {orig_val:<12.4f} {crs_val:<12.4f} {reduction:<11.1f}% {scale:<10.4f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Original absmax (mean): {np.mean(orig_absmax):.4f}")
    print(f"  CRS-fused absmax (mean): {np.mean(crs_absmax):.4f}")
    print(f"  Outlier dimensions: {len(outlier_indices)}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare Key activations between Original and CRS-fused models'
    )
    parser.add_argument('orig_rope_file', help='Original model rope_values_full.bin')
    parser.add_argument('crs_rope_file', help='CRS-fused model rope_values_full.bin')
    parser.add_argument('scales_file', help='CRS scales file')
    parser.add_argument('--layer', type=int, default=0, help='Layer index (default: 0)')
    parser.add_argument('--head', type=int, default=0, help='Head index (default: 0)')
    parser.add_argument('--post-rope', action='store_true', help='Use post-RoPE values')
    parser.add_argument('--output', '-o', help='Output image path')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Key Activation Comparison: Original vs CRS-Fused")
    print("="*70)
    
    layer_idx = args.layer
    head_idx = args.head
    
    print(f"\n[1/4] Loading original model activations (Layer {layer_idx})...")
    orig_data = load_rope_values(args.orig_rope_file, target_layer=layer_idx)
    
    print(f"\n[2/4] Loading CRS-fused model activations (Layer {layer_idx})...")
    crs_data = load_rope_values(args.crs_rope_file, target_layer=layer_idx)
    
    print(f"\n[3/4] Loading CRS scales...")
    scales_data = load_crs_scales(args.scales_file)
    
    # Get activations for specified layer and head
    if args.post_rope:
        orig_acts = orig_data['layer_data']['post'][head_idx]  # [n_tokens, n_dims]
        crs_acts = crs_data['layer_data']['post'][head_idx]
        print(f"\nUsing post-RoPE activations")
    else:
        orig_acts = orig_data['layer_data']['pre'][head_idx]
        crs_acts = crs_data['layer_data']['pre'][head_idx]
        print(f"\nUsing pre-RoPE activations")
    
    print(f"  Original: {orig_acts.shape[0]} tokens")
    print(f"  CRS-fused: {crs_acts.shape[0]} tokens")
    
    # Compute absmax
    print(f"\n[4/4] Computing absmax and generating visualization...")
    orig_absmax = compute_absmax(orig_acts)
    crs_absmax = compute_absmax(crs_acts)
    
    # Get outlier info
    head_scales = scales_data['scales'][layer_idx][head_idx]
    outlier_indices = head_scales['indices']
    outlier_scales = head_scales['scales']
    
    # Visualize
    visualize_comparison(
        orig_absmax, crs_absmax, outlier_indices, outlier_scales,
        layer_idx, head_idx, args.post_rope, args.output
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
