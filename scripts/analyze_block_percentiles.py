#!/usr/bin/env python3
"""
Analyze block-level token-wise statistics for percentile-based clipping.

Extracts per-block statistics:
- p50, p90, p99, p99.9 of token-wise block_absmax
- absmax / p99.9 ratio (clipping potential indicator)
- argmax dim frequency (which dims dominate block absmax)
"""

import struct
import numpy as np
import argparse
from collections import defaultdict
import json

def read_ropv_header(f):
    magic = struct.unpack('I', f.read(4))[0]
    if magic != 0x524F5056:  # "ROPV"
        raise ValueError(f"Invalid magic: 0x{magic:08X}")
    
    version = struct.unpack('I', f.read(4))[0]
    n_layers = struct.unpack('I', f.read(4))[0]
    n_heads = struct.unpack('I', f.read(4))[0]
    n_dims = struct.unpack('I', f.read(4))[0]
    n_tokens = struct.unpack('I', f.read(4))[0]  # uint32, not uint64
    
    return version, n_layers, n_heads, n_dims, n_tokens

def analyze_block_statistics(ropv_file, target_layer=None, dim_min=64, output_json=None):
    """
    Analyze block-level statistics from raw activation dump.
    
    For each (layer, head, block):
    - Calculate token-wise block_absmax
    - Extract percentiles (p50, p90, p99, p99.9)
    - Calculate absmax/p99.9 ratio
    - Track argmax dim frequency
    """
    
    with open(ropv_file, 'rb') as f:
        version, n_layers, n_heads, n_dims, n_tokens = read_ropv_header(f)
        
        print(f"ROPV File: {ropv_file}")
        print(f"  Layers: {n_layers}, Heads: {n_heads}, Dims: {n_dims}, Tokens: {n_tokens}")
        print(f"  Analyzing dims >= {dim_min}")
        print()
        
        # Define blocks (assuming 32-dim blocks for Q4_0)
        blocks = []
        for start in range(dim_min, n_dims, 32):
            end = min(start + 32, n_dims)
            blocks.append((start, end))
        
        print(f"Blocks to analyze: {len(blocks)}")
        for i, (start, end) in enumerate(blocks):
            print(f"  Block {i}: dims [{start}, {end})")
        print()
        
        results = {}
        
        for layer_idx in range(n_layers):
            # Read pre-RoPE count and data
            pre_count = struct.unpack('I', f.read(4))[0]
            
            if target_layer is not None and layer_idx != target_layer:
                # Skip this layer
                if pre_count > 0:
                    f.seek(pre_count * 4, 1)  # Skip pre_values (float32)
                post_count = struct.unpack('I', f.read(4))[0]
                if post_count > 0:
                    f.seek(post_count * 4, 1)  # Skip post_values
                continue
            
            print(f"Processing Layer {layer_idx}...")
            
            if pre_count == 0:
                # No data for this layer
                post_count = struct.unpack('I', f.read(4))[0]
                if post_count > 0:
                    f.seek(post_count * 4, 1)
                continue
            
            # Read pre-RoPE values (float32)
            pre_values = np.frombuffer(f.read(pre_count * 4), dtype=np.float32)
            n_tokens_layer = pre_count // (n_heads * n_dims)
            pre_values = pre_values.reshape(n_tokens_layer, n_heads, n_dims)
            
            # Skip post-RoPE (not needed for this analysis)
            post_count = struct.unpack('I', f.read(4))[0]
            if post_count > 0:
                f.seek(post_count * 4, 1)
            
            # Analyze each head
            for head_idx in range(n_heads):
                head_data = pre_values[:, head_idx, :]  # [n_tokens_layer, n_dims]
                
                # Analyze each block
                for block_idx, (start, end) in enumerate(blocks):
                    block_data = head_data[:, start:end]  # [n_tokens_layer, block_size]
                    
                    # Token-wise block absmax
                    token_block_absmax = np.max(np.abs(block_data), axis=1)  # [n_tokens_layer]
                    
                    # Token-wise argmax dim (within block)
                    token_argmax_local = np.argmax(np.abs(block_data), axis=1)  # [n_tokens_layer]
                    token_argmax_global = token_argmax_local + start
                    
                    # Calculate percentiles
                    p50 = np.percentile(token_block_absmax, 50)
                    p90 = np.percentile(token_block_absmax, 90)
                    p99 = np.percentile(token_block_absmax, 99)
                    p999 = np.percentile(token_block_absmax, 99.9)
                    absmax = np.max(token_block_absmax)
                    
                    # Clipping potential indicator
                    ratio = absmax / p999 if p999 > 1e-6 else 1.0
                    
                    # Argmax dim frequency
                    argmax_counts = defaultdict(int)
                    for dim in token_argmax_global:
                        argmax_counts[int(dim)] += 1
                    
                    # Sort by frequency
                    top_argmax = sorted(argmax_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    # Outlier count (> p99.9)
                    outlier_count = np.sum(token_block_absmax > p999)
                    outlier_ratio = outlier_count / n_tokens_layer
                    
                    key = f"L{layer_idx}_H{head_idx}_B{block_idx}"
                    results[key] = {
                        'layer': layer_idx,
                        'head': head_idx,
                        'block': block_idx,
                        'dim_range': [start, end],
                        'n_tokens': int(n_tokens_layer),
                        'p50': float(p50),
                        'p90': float(p90),
                        'p99': float(p99),
                        'p99.9': float(p999),
                        'absmax': float(absmax),
                        'absmax/p99.9': float(ratio),
                        'outlier_count': int(outlier_count),
                        'outlier_ratio': float(outlier_ratio),
                        'top_argmax_dims': [(int(d), int(c)) for d, c in top_argmax]
                    }
        
        # Sort by absmax/p99.9 ratio (descending)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['absmax/p99.9'], reverse=True)
        
        print("=" * 100)
        print("Block-Level Statistics (sorted by absmax/p99.9 ratio)")
        print("=" * 100)
        print(f"{'Key':<15} {'p50':<8} {'p90':<8} {'p99':<8} {'p99.9':<8} {'absmax':<8} {'ratio':<8} {'outlier%':<10} {'top_argmax_dim':<20}")
        print("-" * 100)
        
        for key, stats in sorted_results[:20]:  # Top 20
            top_dim = stats['top_argmax_dims'][0] if stats['top_argmax_dims'] else (0, 0)
            print(f"{key:<15} {stats['p50']:<8.3f} {stats['p90']:<8.3f} {stats['p99']:<8.3f} "
                  f"{stats['p99.9']:<8.3f} {stats['absmax']:<8.3f} {stats['absmax/p99.9']:<8.3f} "
                  f"{stats['outlier_ratio']*100:<9.4f}% dim{top_dim[0]}({top_dim[1]})")
        
        print("=" * 100)
        print()
        
        # Summary statistics
        high_ratio_blocks = [k for k, v in results.items() if v['absmax/p99.9'] > 1.5]
        rare_outliers = [k for k, v in results.items() if v['outlier_ratio'] < 0.001]
        candidates = set(high_ratio_blocks) & set(rare_outliers)
        
        print(f"Summary:")
        print(f"  Total blocks analyzed: {len(results)}")
        print(f"  High ratio (>1.5) blocks: {len(high_ratio_blocks)} ({len(high_ratio_blocks)/len(results)*100:.1f}%)")
        print(f"  Rare outliers (<0.1%) blocks: {len(rare_outliers)} ({len(rare_outliers)/len(results)*100:.1f}%)")
        print(f"  Clipping candidates (both): {len(candidates)} ({len(candidates)/len(results)*100:.1f}%)")
        print()
        
        if output_json:
            with open(output_json, 'w') as jf:
                json.dump({
                    'metadata': {
                        'n_layers': n_layers,
                        'n_heads': n_heads,
                        'n_dims': n_dims,
                        'n_tokens': int(n_tokens),
                        'dim_min': dim_min
                    },
                    'blocks': results,
                    'summary': {
                        'high_ratio_blocks': high_ratio_blocks,
                        'rare_outliers': rare_outliers,
                        'clipping_candidates': list(candidates)
                    }
                }, jf, indent=2)
            print(f"Results saved to: {output_json}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Analyze block-level percentile statistics')
    parser.add_argument('ropv_file', help='Input ROPV file (raw activation dump)')
    parser.add_argument('--layer', type=int, help='Target layer (analyze all if not specified)')
    parser.add_argument('--dim-min', type=int, default=64, help='Minimum dimension (default: 64)')
    parser.add_argument('--output', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    analyze_block_statistics(args.ropv_file, args.layer, args.dim_min, args.output)

if __name__ == '__main__':
    main()
