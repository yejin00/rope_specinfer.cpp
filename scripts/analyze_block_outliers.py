#!/usr/bin/env python3
import argparse
import numpy as np
import os
import struct

def read_activation_dump(file_path, n_head_kv=8, head_dim=128):
    """
    Reads binary dump of activation (Key) tensor.
    Shape: [n_tokens, n_head_kv, head_dim]
    """
    file_size = os.path.getsize(file_path)
    n_elements = file_size // 4
    elems_per_token = n_head_kv * head_dim
    n_tokens = n_elements // elems_per_token
    
    print(f"Loading {file_path}...")
    print(f"  Tokens: {n_tokens}, Heads: {n_head_kv}, Dim: {head_dim}")
    
    data = np.fromfile(file_path, dtype=np.float32)
    try:
        data = data.reshape(n_tokens, n_head_kv, head_dim)
    except ValueError:
        print(f"Error: Shape mismatch. Size {data.size} vs {n_tokens}x{n_head_kv}x{head_dim}")
        return None
    return data

def analyze_blocks(data, layer_idx, block_size=32):
    n_tokens, n_heads, head_dim = data.shape
    n_blocks = head_dim // block_size
    
    print(f"\nAnalyzing Layer {layer_idx} (Block Size: {block_size})")
    print(f"{'Head':<5} {'Block':<6} {'Max':<8} {'p99.9':<8} {'Ratio':<6} {'Dominant Dim (Freq%)':<30}")
    print("-" * 80)
    
    # Iterate over heads and blocks
    for h in range(n_heads):
        for b in range(n_blocks):
            # Extract block data: [n_tokens, block_size]
            start_dim = b * block_size
            end_dim = start_dim + block_size
            block_data = data[:, h, start_dim:end_dim]
            
            # Compute absolute values
            abs_data = np.abs(block_data)
            
            # 1. Block AbsMax per token: [n_tokens]
            block_absmax = np.max(abs_data, axis=1)
            
            # 2. Argmax Dim per token (relative to block start): [n_tokens]
            argmax_indices = np.argmax(abs_data, axis=1)
            
            # Statistics
            max_val = np.max(block_absmax)
            p99_9 = np.percentile(block_absmax, 99.9)
            p99 = np.percentile(block_absmax, 99.0)
            p50 = np.median(block_absmax)
            
            # Ratio (Gain if we clip outliers)
            ratio = max_val / p99_9 if p99_9 > 1e-6 else 1.0
            
            # Dominant Dimensions
            # Count frequency of each dim being the max
            counts = np.bincount(argmax_indices, minlength=block_size)
            total_counts = np.sum(counts)
            
            # Find top dominant dim
            dom_idx_local = np.argmax(counts)
            dom_freq = counts[dom_idx_local] / total_counts * 100
            dom_idx_global = start_dim + dom_idx_local
            
            # Only print interesting blocks (high ratio or specific request)
            # Filter: Ratio > 1.2 (20% potential gain) OR user wants to see everything?
            # Let's print top blocks sorted by ratio later? 
            # For now, print if ratio > 1.1 to reduce noise, or just print all compact.
            
            dom_str = f"Dim {dom_idx_global} ({dom_freq:.1f}%)"
            
            # If there's a second dominant one
            counts[dom_idx_local] = 0
            if np.max(counts) > 0:
                dom2_idx = np.argmax(counts)
                dom2_freq = counts[dom2_idx] / total_counts * 100
                if dom2_freq > 10.0: # Only show if significant
                    dom_str += f", Dim {start_dim + dom2_idx} ({dom2_freq:.1f}%)"
            
            # Highlight high ratios
            marker = "*" if ratio > 1.5 else " "
            
            print(f"{h:<5} {b:<6} {max_val:<8.4f} {p99_9:<8.4f} {ratio:<6.2f}{marker} {dom_str}")

def main():
    parser = argparse.ArgumentParser(description='Analyze block-wise outlier statistics')
    parser.add_argument('file_path', help='Path to activation_layer_X.bin')
    parser.add_argument('--layer', type=int, required=True, help='Layer index')
    parser.add_argument('--block-size', type=int, default=32, help='Quantization block size')
    
    args = parser.parse_args()
    
    data = read_activation_dump(args.file_path)
    if data is not None:
        analyze_blocks(data, args.layer, args.block_size)

if __name__ == '__main__':
    main()
