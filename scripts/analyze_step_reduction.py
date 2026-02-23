#!/usr/bin/env python3
"""
Analyze Step Reduction: Calculate how much PRS reduces block absmax (quantization step)

For each token:
  m(t) = max_j |x_j(t)|  (original block absmax)
  m'(t) = max_j |x_j(t)| / s_j  (PRS-scaled block absmax, selected dims only)
  
  Step reduction ratio: r(t) = m'(t) / m(t)
  
Smaller r(t) means better quantization (smaller step)
"""

import struct
import numpy as np
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

def load_prs_scales(scales_path):
    """Load PRS scales and return per-block scale mapping"""
    with open(scales_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != 0x53435253:  # 'SRCS'
            raise ValueError(f"Invalid magic: 0x{magic:08X}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        top_k = struct.unpack('I', f.read(4))[0]
        
        n_outliers = top_k * 2  # pairs
        
        print(f"Loading PRS scales: layers={n_layers}, heads={n_heads}, dims={n_dims}, k={top_k}")
        
        # Store scales as: scales_map[(layer, head, dim)] = scale
        scales_map = {}
        
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                indices = np.frombuffer(f.read(n_outliers * 4), dtype=np.int32)
                scales = np.frombuffer(f.read(n_outliers * 4), dtype=np.float32)
                
                for dim, scale in zip(indices, scales):
                    if dim >= 0 and scale > 0:
                        scales_map[(layer_idx, head_idx, int(dim))] = float(scale)
        
        return scales_map, n_layers, n_heads

def read_layer_dump(layer_file, n_heads=8, n_dims=128):
    """Read a single layer activation dump"""
    file_size = os.path.getsize(layer_file)
    n_elements = file_size // 4
    elems_per_token = n_heads * n_dims
    n_tokens = n_elements // elems_per_token
    
    data = np.fromfile(layer_file, dtype=np.float32)
    try:
        data = data.reshape(n_tokens, n_heads, n_dims)
        return data, n_tokens
    except ValueError:
        return None, 0

def calculate_step_reduction(layer_dir_prefix, scales_map, dim_min=64, n_heads=8, n_dims=128):
    """
    Calculate per-token step reduction for all blocks
    
    Returns: dict of results per block
    """
    prefix_path = Path(layer_dir_prefix)
    layer_dir = prefix_path.parent
    file_prefix = prefix_path.name
    
    layer_files = sorted(layer_dir.glob(f"{file_prefix}_layer_*.bin"))
    
    print(f"\nFound {len(layer_files)} layer files")
    print(f"Analyzing dims >= {dim_min}")
    
    # Define blocks
    blocks = []
    for start in range(dim_min, n_dims, 32):
        end = min(start + 32, n_dims)
        blocks.append((start, end))
    
    print(f"Blocks: {len(blocks)}")
    for i, (start, end) in enumerate(blocks):
        print(f"  Block {i}: [{start}, {end})")
    
    results = {}
    
    for layer_file in layer_files:
        filename = layer_file.name
        layer_idx = int(filename.split('_layer_')[1].split('.')[0])
        
        print(f"\nProcessing Layer {layer_idx}...")
        
        layer_data, n_tokens = read_layer_dump(layer_file, n_heads, n_dims)
        if layer_data is None or n_tokens == 0:
            continue
        
        for head_idx in range(n_heads):
            head_data = layer_data[:, head_idx, :]  # [n_tokens, n_dims]
            
            for block_idx, (start, end) in enumerate(blocks):
                block_data = head_data[:, start:end]  # [n_tokens, block_size]
                
                # Calculate per-token step reduction
                ratios = []
                
                for token_idx in range(n_tokens):
                    token_block = block_data[token_idx]  # [block_size]
                    
                    # Original block absmax
                    m_orig = np.max(np.abs(token_block))
                    
                    if m_orig < 1e-8:
                        ratios.append(1.0)
                        continue
                    
                    # PRS-scaled block absmax
                    scaled_values = []
                    for local_dim, val in enumerate(token_block):
                        global_dim = start + local_dim
                        scale = scales_map.get((layer_idx, head_idx, global_dim), 1.0)
                        scaled_values.append(abs(val) / scale)
                    
                    m_scaled = max(scaled_values)
                    
                    # Step reduction ratio
                    ratio = m_scaled / m_orig if m_orig > 0 else 1.0
                    ratios.append(ratio)
                
                # Statistics
                ratios = np.array(ratios)
                
                key = f"L{layer_idx}_H{head_idx}_B{block_idx}"
                results[key] = {
                    'layer': layer_idx,
                    'head': head_idx,
                    'block': block_idx,
                    'dim_range': [start, end],
                    'n_tokens': int(n_tokens),
                    'mean_ratio': float(np.mean(ratios)),
                    'median_ratio': float(np.median(ratios)),
                    'p10_ratio': float(np.percentile(ratios, 10)),
                    'p90_ratio': float(np.percentile(ratios, 90)),
                    'min_ratio': float(np.min(ratios)),
                    'max_ratio': float(np.max(ratios)),
                    'step_reduction_pct': float((1.0 - np.mean(ratios)) * 100)
                }
    
    return results

def print_summary(results):
    """Print step reduction summary"""
    print("\n" + "=" * 100)
    print("Step Reduction Analysis (K+5 PRS)")
    print("=" * 100)
    
    # Sort by step reduction percentage (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['step_reduction_pct'], reverse=True)
    
    print(f"\n{'Key':<15} {'Mean r(t)':<12} {'Median':<10} {'p10':<10} {'p90':<10} {'Step ↓%':<10}")
    print("-" * 100)
    
    for key, stats in sorted_results[:20]:
        print(f"{key:<15} {stats['mean_ratio']:<12.4f} {stats['median_ratio']:<10.4f} "
              f"{stats['p10_ratio']:<10.4f} {stats['p90_ratio']:<10.4f} {stats['step_reduction_pct']:<10.2f}%")
    
    print("\n" + "=" * 100)
    
    # Overall statistics
    all_mean_ratios = [r['mean_ratio'] for r in results.values()]
    all_step_reductions = [r['step_reduction_pct'] for r in results.values()]
    
    print(f"\nOverall Statistics (across {len(results)} blocks):")
    print(f"  Average step reduction: {np.mean(all_step_reductions):.2f}%")
    print(f"  Median step reduction:  {np.median(all_step_reductions):.2f}%")
    print(f"  Best block:             {np.max(all_step_reductions):.2f}%")
    print(f"  Worst block:            {np.min(all_step_reductions):.2f}%")
    print()
    print(f"  Average r(t):           {np.mean(all_mean_ratios):.4f}")
    print(f"  Median r(t):            {np.median(all_mean_ratios):.4f}")
    
    # Blocks by effectiveness
    strong = [k for k, v in results.items() if v['step_reduction_pct'] > 10]
    moderate = [k for k, v in results.items() if 5 < v['step_reduction_pct'] <= 10]
    weak = [k for k, v in results.items() if v['step_reduction_pct'] <= 5]
    
    print(f"\nBlocks by Step Reduction Effectiveness:")
    print(f"  Strong (>10%):     {len(strong)} ({len(strong)/len(results)*100:.1f}%)")
    print(f"  Moderate (5-10%):  {len(moderate)} ({len(moderate)/len(results)*100:.1f}%)")
    print(f"  Weak (≤5%):        {len(weak)} ({len(weak)/len(results)*100:.1f}%)")
    
    print("=" * 100)

def main():
    parser = argparse.ArgumentParser(description='Analyze step reduction from PRS')
    parser.add_argument('layer_prefix', help='Layer dump prefix (e.g., /path/to/dump/activation)')
    parser.add_argument('scales_file', help='PRS scales binary file')
    parser.add_argument('--dim-min', type=int, default=64, help='Minimum dimension')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-dims', type=int, default=128)
    
    args = parser.parse_args()
    
    # Load scales
    print("Loading PRS scales...")
    scales_map, n_layers, n_heads = load_prs_scales(args.scales_file)
    print(f"Loaded {len(scales_map)} scale entries")
    
    # Calculate step reduction
    results = calculate_step_reduction(
        args.layer_prefix,
        scales_map,
        args.dim_min,
        args.n_heads,
        args.n_dims
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
