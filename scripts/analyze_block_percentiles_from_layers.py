#!/usr/bin/env python3
"""
Analyze block-level token-wise statistics from layer dump files directly.
Avoids merging to save disk space.
"""

import struct
import numpy as np
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

def read_layer_dump(layer_file, n_heads=8, n_dims=128):
    """
    Read a single layer activation dump file.
    Returns: [n_tokens, n_heads, n_dims] array
    """
    file_size = os.path.getsize(layer_file)
    n_elements = file_size // 4  # float32
    
    elems_per_token = n_heads * n_dims
    n_tokens = n_elements // elems_per_token
    
    # Read data
    data = np.fromfile(layer_file, dtype=np.float32)
    
    # Reshape to [n_tokens, n_heads, n_dims]
    try:
        data = data.reshape(n_tokens, n_heads, n_dims)
        return data, n_tokens
    except ValueError:
        print(f"Warning: Could not reshape {layer_file}")
        return None, 0

def analyze_block_statistics_from_layers(layer_dir_prefix, target_layer=None, dim_min=64, output_json=None, n_heads=8, n_dims=128):
    """
    Analyze block-level statistics from layer dump files.
    
    layer_dir_prefix: path prefix like "/path/to/dump/activation"
    Expects files: activation_layer_0.bin, activation_layer_1.bin, ...
    """
    
    # Find all layer files
    prefix_path = Path(layer_dir_prefix)
    layer_dir = prefix_path.parent
    file_prefix = prefix_path.name
    
    layer_files = sorted(layer_dir.glob(f"{file_prefix}_layer_*.bin"))
    
    print(f"Found {len(layer_files)} layer files")
    print(f"Analyzing dims >= {dim_min}")
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
    
    for layer_file in layer_files:
        # Extract layer index from filename
        filename = layer_file.name
        layer_idx = int(filename.split('_layer_')[1].split('.')[0])
        
        if target_layer is not None and layer_idx != target_layer:
            continue
        
        print(f"Processing Layer {layer_idx}...")
        
        # Read layer data
        layer_data, n_tokens = read_layer_dump(layer_file, n_heads, n_dims)
        
        if layer_data is None or n_tokens == 0:
            continue
        
        # Analyze each head
        for head_idx in range(n_heads):
            head_data = layer_data[:, head_idx, :]  # [n_tokens, n_dims]
            
            # Analyze each block
            for block_idx, (start, end) in enumerate(blocks):
                block_data = head_data[:, start:end]  # [n_tokens, block_size]
                
                # Token-wise block absmax
                token_block_absmax = np.max(np.abs(block_data), axis=1)  # [n_tokens]
                
                # Token-wise argmax dim (within block)
                token_argmax_local = np.argmax(np.abs(block_data), axis=1)  # [n_tokens]
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
                
                # Sort by frequency (store top 16 for fair comparison with PRS)
                top_argmax = sorted(argmax_counts.items(), key=lambda x: x[1], reverse=True)[:16]
                
                # Outlier count (> p99.9)
                outlier_count = np.sum(token_block_absmax > p999)
                outlier_ratio = outlier_count / n_tokens
                
                key = f"L{layer_idx}_H{head_idx}_B{block_idx}"
                results[key] = {
                    'layer': layer_idx,
                    'head': head_idx,
                    'block': block_idx,
                    'dim_range': [start, end],
                    'n_tokens': int(n_tokens),
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
        # Get sample n_tokens (should be consistent across layers)
        sample_n_tokens = results[list(results.keys())[0]]['n_tokens'] if results else 0
        
        with open(output_json, 'w') as jf:
            json.dump({
                'metadata': {
                    'n_layers': len(layer_files),
                    'n_heads': n_heads,
                    'n_dims': n_dims,
                    'n_tokens': int(sample_n_tokens),
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
    parser = argparse.ArgumentParser(description='Analyze block-level percentile statistics from layer dumps')
    parser.add_argument('layer_prefix', help='Layer file prefix (e.g., /path/to/dump/activation)')
    parser.add_argument('--layer', type=int, help='Target layer (analyze all if not specified)')
    parser.add_argument('--dim-min', type=int, default=64, help='Minimum dimension (default: 64)')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--n-dims', type=int, default=128, help='Number of dimensions')
    
    args = parser.parse_args()
    
    analyze_block_statistics_from_layers(
        args.layer_prefix,
        args.layer,
        args.dim_min,
        args.output,
        args.n_heads,
        args.n_dims
    )

if __name__ == '__main__':
    main()
