#!/usr/bin/env python3
"""
Plot block-level absmax histogram for clipping analysis.

For a given (layer, head, block):
- Extract all tokens' activations in that block [n_tokens, 32]
- Calculate token-wise block_absmax = max(abs(block_values), axis=1)
- Plot histogram of these block_absmax values
- Mark percentiles (p99, p99.9) and absmax for clipping analysis
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import struct

MAGIC_ROPV = 0x524F5056  # "ROPV"

def read_block_data(file_path, target_layer, target_head, block_start, block_end):
    """
    Read activation data for a specific block.
    Returns: [n_tokens, block_size] array
    """
    with open(file_path, 'rb') as f:
        # Read header
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: 0x{magic:08X}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        n_tokens = struct.unpack('I', f.read(4))[0]
        
        print(f"ROPV format: layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")
        
        # Read through layers
        for layer_idx in range(n_layers):
            pre_count = struct.unpack('I', f.read(4))[0]
            
            if layer_idx == target_layer:
                if pre_count == 0:
                    print(f"Layer {layer_idx}: No data")
                    return None, 0
                
                # Read pre-RoPE values (float32)
                pre_values = np.frombuffer(f.read(pre_count * 4), dtype=np.float32)
                n_tokens_layer = pre_count // (n_heads * n_dims)
                pre_values = pre_values.reshape(n_tokens_layer, n_heads, n_dims)
                
                # Extract target head and block
                head_data = pre_values[:, target_head, :]  # [n_tokens, n_dims]
                block_data = head_data[:, block_start:block_end]  # [n_tokens, block_size]
                
                # Skip post-RoPE
                post_count = struct.unpack('I', f.read(4))[0]
                if post_count > 0:
                    f.seek(post_count * 4, 1)
                
                print(f"Layer {layer_idx} Head {target_head} Block [{block_start}:{block_end}]: {n_tokens_layer} tokens")
                return block_data, n_tokens_layer
            else:
                # Skip this layer
                if pre_count > 0:
                    f.seek(pre_count * 4, 1)
                post_count = struct.unpack('I', f.read(4))[0]
                if post_count > 0:
                    f.seek(post_count * 4, 1)
        
        return None, 0

def plot_block_histogram(block_data, layer_idx, head_idx, block_idx, block_range, output_path):
    """
    Plot histogram of token-wise block absmax.
    
    X-axis: Block absmax value (max absolute value across 32 dims per token)
    Y-axis: Token count (log scale)
    """
    # Calculate token-wise block absmax
    token_block_absmax = np.max(np.abs(block_data), axis=1)  # [n_tokens]
    
    # Statistics
    n_tokens = len(token_block_absmax)
    p50 = np.percentile(token_block_absmax, 50)
    p90 = np.percentile(token_block_absmax, 90)
    p99 = np.percentile(token_block_absmax, 99)
    p999 = np.percentile(token_block_absmax, 99.9)
    absmax = np.max(token_block_absmax)
    mean_val = np.mean(token_block_absmax)
    
    # Clipping potential
    ratio = absmax / p999 if p999 > 1e-6 else 1.0
    outlier_count = np.sum(token_block_absmax > p999)
    outlier_ratio = outlier_count / n_tokens
    
    print(f"\n[Block Statistics]")
    print(f"  Block: L{layer_idx} H{head_idx} B{block_idx} (dims {block_range[0]}-{block_range[1]-1})")
    print(f"  Tokens: {n_tokens}")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  p50: {p50:.4f}")
    print(f"  p90: {p90:.4f}")
    print(f"  p99: {p99:.4f}")
    print(f"  p99.9: {p999:.4f}")
    print(f"  absmax: {absmax:.4f}")
    print(f"  absmax/p99.9: {ratio:.4f}")
    print(f"  Outliers (>p99.9): {outlier_count} ({outlier_ratio*100:.4f}%)")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Histogram
    counts, bins, patches = ax.hist(token_block_absmax, bins=100, color='skyblue', 
                                      edgecolor='black', alpha=0.7)
    ax.set_yscale('log')
    
    # Mark percentiles
    ax.axvline(p99, color='orange', linestyle='--', linewidth=2, label=f'p99 = {p99:.2f}')
    ax.axvline(p999, color='red', linestyle='--', linewidth=2, label=f'p99.9 = {p999:.2f}')
    ax.axvline(absmax, color='darkred', linestyle='-', linewidth=2, label=f'absmax = {absmax:.2f}')
    
    # Labels and title
    ax.set_xlabel('Block Absmax (max of abs values across 32 dims per token)', fontsize=12)
    ax.set_ylabel('Token Count (log scale)', fontsize=12)
    ax.set_title(f'Block-Level Absmax Distribution\n'
                 f'Layer {layer_idx}, Head {head_idx}, Block {block_idx} (dims {block_range[0]}-{block_range[1]-1})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    # Stats box
    stats_text = (f'Tokens: {n_tokens}\n'
                  f'Mean: {mean_val:.2f}\n'
                  f'p99.9: {p999:.2f}\n'
                  f'absmax: {absmax:.2f}\n'
                  f'Ratio: {ratio:.3f}\n'
                  f'Outliers: {outlier_ratio*100:.3f}%')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    # Clipping potential annotation
    if ratio > 1.5:
        potential = 'Strong Clipping Potential'
        color = 'green'
    elif ratio > 1.3:
        potential = 'Moderate Clipping Potential'
        color = 'orange'
    else:
        potential = 'Weak Clipping Potential'
        color = 'gray'
    
    ax.text(0.98, 0.02, potential, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
            fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot block-level absmax histogram')
    parser.add_argument('file_path', help='Path to ROPV file')
    parser.add_argument('--layer', type=int, required=True, help='Layer index')
    parser.add_argument('--head', type=int, required=True, help='Head index (0-7)')
    parser.add_argument('--block', type=int, required=True, help='Block index (0=dims[0:32], 1=dims[32:64], etc)')
    parser.add_argument('--output', '-o', required=True, help='Output image path')
    
    args = parser.parse_args()
    
    # Calculate block range (assuming 32-dim blocks)
    block_start = args.block * 32
    block_end = block_start + 32
    
    block_data, n_tokens = read_block_data(args.file_path, args.layer, args.head, 
                                            block_start, block_end)
    
    if block_data is not None and n_tokens > 0:
        plot_block_histogram(block_data, args.layer, args.head, args.block,
                            (block_start, block_end), args.output)

if __name__ == '__main__':
    main()
