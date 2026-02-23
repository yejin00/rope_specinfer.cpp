#!/usr/bin/env python3
"""
Extract Static CRS Scales from rope_absmax_values.bin using Block-Aware Greedy Strategy

This file format (APOR magic):
- Magic: 0x524F5041 ("APOR")
- Version: uint32
- n_layers: uint32
- n_heads: uint32
- n_dims: uint32
- For each layer, for each head:
    - pre_absmax[n_dims]: float32  (Pre-RoPE channel-wise absmax)
    - post_absmax[n_dims]: float32 (Post-RoPE channel-wise absmax)

Output: scales_static_k.bin for CRS
"""

import struct
import numpy as np
import argparse
import sys
from pathlib import Path

MAGIC_APOR = 0x524F5041  # "APOR"
MAGIC_SCRS = 0x53435253  # "SCRS" - Static CRS
VERSION = 1


def read_absmax_values(file_path):
    """Read rope_absmax_values.bin file"""
    import os
    file_size = os.path.getsize(file_path)
    
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_APOR:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected {hex(MAGIC_APOR)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        
        # Check actual layers in file
        bytes_per_layer = n_heads * n_dims * 4 * 2  # pre + post
        actual_layers = (file_size - 20) // bytes_per_layer
        
        if actual_layers < n_layers:
            print(f"WARNING: File has {actual_layers} layers, header says {n_layers}")
            n_layers = actual_layers
        
        print(f"Absmax values file:")
        print(f"  Version: {version}")
        print(f"  Layers: {n_layers}, Heads: {n_heads}, Dims: {n_dims}")
        
        data = {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'pre_absmax': [],   # [layer][head][dim]
            'post_absmax': []   # [layer][head][dim]
        }
        
        for layer in range(n_layers):
            layer_pre = []
            layer_post = []
            for head in range(n_heads):
                pre = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                post = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                layer_pre.append(pre)
                layer_post.append(post)
            data['pre_absmax'].append(layer_pre)
            data['post_absmax'].append(layer_post)
        
        return data


def analyze_beta(data, top_k=8, dim_min=80):
    """Analyze optimal beta by computing outlier/median ratios with PRS pair logic"""
    n_layers = data['n_layers']
    n_heads = data['n_heads']
    
    all_ratios = []
    
    print(f"\n=== Analyzing optimal β (PRS-aware) ===")
    print(f"Computing outlier/median ratios with RoPE pair exclusion...")
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            pre_absmax = data['pre_absmax'][layer_idx][head_idx]
            
            # Find top-k outliers (all dims)
            top_indices = np.argsort(pre_absmax)[-top_k:][::-1]
            top_scales = pre_absmax[top_indices]
            
            # Filter to dim >= dim_min
            mask = top_indices >= dim_min
            filtered_indices = top_indices[mask]
            filtered_scales = top_scales[mask]
            
            if len(filtered_indices) == 0:
                continue
            
            # Convert to RoPE pairs (same logic as extract_static_outliers)
            pair_set = set()
            pair_scales = {}
            
            for idx, scale in zip(filtered_indices, filtered_scales):
                d_even = int(idx & ~1)
                d_odd = int(idx | 1)
                pair_key = (d_even, d_odd)
                
                if pair_key not in pair_set:
                    pair_set.add(pair_key)
                    pair_scale = max(pre_absmax[d_even], pre_absmax[d_odd])
                    pair_scales[pair_key] = pair_scale
            
            # Collect all PRS dimensions (both even and odd in pairs)
            prs_dims = set()
            for (d_even, d_odd) in pair_set:
                prs_dims.add(d_even)
                prs_dims.add(d_odd)
            
            # Calculate median of non-PRS channels (ALL dims, not in PRS pairs)
            non_prs_mask = np.ones(len(pre_absmax), dtype=bool)
            for d in prs_dims:
                if d < len(pre_absmax):
                    non_prs_mask[d] = False
            
            if np.sum(non_prs_mask) == 0:
                continue
            
            median_normal = np.median(pre_absmax[non_prs_mask])
            
            # Compute ratios for each pair (using pair's max absmax)
            for pair_absmax in pair_scales.values():
                ratio = pair_absmax / median_normal if median_normal > 0 else 1.0
                all_ratios.append(ratio)
    
    all_ratios = np.array(all_ratios)
    
    print(f"\nβ analysis results (PRS pairs):")
    print(f"  Total PRS pairs analyzed: {len(all_ratios)}")
    print(f"  Ratio (pair_absmax / median_non_prs) statistics:")
    print(f"    Mean:   {np.mean(all_ratios):.2f}")
    print(f"    Median: {np.median(all_ratios):.2f}")
    print(f"    Std:    {np.std(all_ratios):.2f}")
    print(f"    Min:    {np.min(all_ratios):.2f}")
    print(f"    Max:    {np.max(all_ratios):.2f}")
    
    # Percentiles
    print(f"\n  Percentiles:")
    for p in [25, 50, 75, 90, 95]:
        print(f"    {p}th: {np.percentile(all_ratios, p):.2f}")
    
    suggested_beta = np.median(all_ratios)
    print(f"\n  Suggested β: {suggested_beta:.2f}")
    print(f"  (Using median to balance outlier suppression)\n")
    
    return suggested_beta


def extract_static_outliers(data, top_k=2, dim_min=80, beta=4.0, max_scale=None, alpha=1.0, verbose=True):
    """
    Extract static outlier channels using TRUE Per-Block Strategy.
    
    Args:
        max_scale: Maximum allowed scale factor. If None, no limit.
        alpha: Power factor for scale damping (scale = raw_scale ** alpha).
               alpha < 1.0 dampens large scales more than small ones.
    """
    n_layers = data['n_layers']
    n_heads = data['n_heads']
    n_dims = data['n_dims']
    BLOCK_SIZE = 32
    TARGET_VAL = beta
    
    # Statistics tracking
    all_scales_raw = []
    
    # Calculate which blocks are eligible
    first_block = dim_min // BLOCK_SIZE
    n_blocks = (n_dims + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Max outliers = top_k pairs * 2 dims/pair * eligible_blocks
    eligible_blocks = n_blocks - first_block
    max_outliers = top_k * 2 * eligible_blocks
    
    results = {
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_dims': n_dims,
        'top_k': max_outliers // 2,  # Store total pairs for header
        'dim_min': dim_min,
        'beta': beta,
        'alpha': alpha,
        'outliers': []
    }
    
    print(f"\n=== TRUE Per-Block Outlier Selection ===")
    print(f"Strategy: {top_k} pairs per block (dims >= {dim_min})")
    print(f"Blocks: {first_block} to {n_blocks-1} ({eligible_blocks} blocks)")
    print(f"Total outliers: {max_outliers} dims ({max_outliers//2} pairs)")
    if alpha != 1.0:
        print(f"Alpha (Damping): {alpha:.2f}")
    if max_scale:
        print(f"Max Scale Limit: {max_scale:.2f}")
    
    for layer_idx in range(n_layers):
        layer_outliers = []
        
        for head_idx in range(n_heads):
            original_absmax = data['pre_absmax'][layer_idx][head_idx]
            
            all_pairs = []
            prs_indices = []
            prs_scales = []
            
            # Process each eligible block
            for block_idx in range(first_block, n_blocks):
                b_start = block_idx * BLOCK_SIZE
                b_end = min(b_start + BLOCK_SIZE, n_dims)
                
                # Get absmax values for this block
                block_absmax = original_absmax[b_start:b_end]
                
                # Find top_k pairs within this block
                # Convert to pairs (even, odd)
                pair_maxes = []
                for i in range(0, len(block_absmax), 2):
                    if i + 1 < len(block_absmax):
                        d_even = b_start + i
                        d_odd = b_start + i + 1
                        pair_max = max(block_absmax[i], block_absmax[i+1])
                        # Store local indices (i, i+1) for exclusion logic
                        pair_maxes.append((pair_max, d_even, d_odd, i, i+1))
                
                # Sort by pair_max and take top_k
                pair_maxes.sort(reverse=True, key=lambda x: x[0])
                
                # Select top_k pairs
                selected_pairs = pair_maxes[:min(top_k, len(pair_maxes))]
                
                # The target is the (k+5)-th pair max (or the last available if len < k+5)
                if len(pair_maxes) > top_k + 5:
                    target_val = pair_maxes[top_k + 5][0]
                elif len(pair_maxes) > top_k:
                    target_val = pair_maxes[-1][0]
                else:
                    target_val = 1.0
                    
                if target_val < 1e-6:
                    target_val = 1.0
                
                # Add selected pairs to total lists
                for p in selected_pairs:
                    all_pairs.append(p)
                    prs_indices.extend([p[1], p[2]])  # even and odd dims
                    
                    pair_absmax = p[0]
                    # Calculate scale needed to bring this outlier down to 'target_val'
                    raw_scale = pair_absmax / target_val
                    final_scale = max(1.0, raw_scale)
                    
                    all_scales_raw.append(final_scale)
                    prs_scales.extend([final_scale, final_scale])
            
            prs_indices = np.array(prs_indices, dtype=np.int32)
            prs_scales = np.array(prs_scales, dtype=np.float32)
            
            # Pad to max_outliers with -1 (invalid)
            if len(prs_indices) < max_outliers:
                pad_len = max_outliers - len(prs_indices)
                prs_indices = np.concatenate([prs_indices, np.full(pad_len, -1, dtype=np.int32)])
                prs_scales = np.concatenate([prs_scales, np.ones(pad_len, dtype=np.float32)])
            
            layer_outliers.append({
                'indices': prs_indices,
                'scales': prs_scales
            })
            
            if verbose and layer_idx % 8 == 0 and head_idx == 0:
                print(f"  Layer {layer_idx}, Head {head_idx}:")
                print(f"    Selected {len(all_pairs)} pairs across {eligible_blocks} blocks")
                # Show first few pairs per block
                for block_idx in range(first_block, min(first_block + 2, n_blocks)):
                    block_pairs = [p for p in all_pairs if block_idx * BLOCK_SIZE <= p[0] < (block_idx + 1) * BLOCK_SIZE]
                    if block_pairs:
                        pairs_str = ", ".join([f"({p[0]},{p[1]})" for p in block_pairs[:3]])
                        print(f"      Block {block_idx} [{block_idx*BLOCK_SIZE}-{(block_idx+1)*BLOCK_SIZE-1}]: {pairs_str}")
        
        results['outliers'].append(layer_outliers)
    
    # Print Scale Stats
    all_scales_raw = np.array(all_scales_raw)
    print(f"\n[Scale Statistics]")
    print(f"  Count: {len(all_scales_raw)}")
    print(f"  Min: {all_scales_raw.min():.4f}")
    print(f"  Max: {all_scales_raw.max():.4f}")
    print(f"  Mean: {all_scales_raw.mean():.4f}")
    print(f"  Median: {np.median(all_scales_raw):.4f}")
    if max_scale:
        n_clipped = np.sum(all_scales_raw > max_scale)
        print(f"  Clipped: {n_clipped} ({n_clipped/len(all_scales_raw)*100:.1f}%) > {max_scale}")

    return results


def analyze_distribution(data, results):
    """Analyze outlier distribution across dimensions"""
    print("\n" + "="*60)
    print("Static Outlier Distribution Analysis")
    print("="*60)
    
    all_indices = []
    for layer_outliers in results['outliers']:
        for head_outliers in layer_outliers:
            all_indices.extend(head_outliers['indices'].tolist())
    
    all_indices = np.array(all_indices)
    n_dims = results['n_dims']
    
    early = np.sum((all_indices >= 0) & (all_indices < 32))
    mid_early = np.sum((all_indices >= 32) & (all_indices < 64))
    mid_late = np.sum((all_indices >= 64) & (all_indices < 96))
    late = np.sum((all_indices >= 96) & (all_indices < n_dims))
    
    total = len(all_indices)
    print(f"\nChannel distribution (total {total} outlier selections):")
    print(f"  Early dims   [0-31]:   {early:4d} ({100*early/total:.1f}%) - High freq (RoPE sensitive)")
    print(f"  Mid-early   [32-63]:   {mid_early:4d} ({100*mid_early/total:.1f}%)")
    print(f"  Mid-late    [64-95]:   {mid_late:4d} ({100*mid_late/total:.1f}%)")
    print(f"  Late dims  [96-127]:   {late:4d} ({100*late/total:.1f}%) - Low freq (RoPE stable)")
    
    # Additional 3-way split: 0-40, 40-80, 80-128
    print(f"\nAlternative 3-way split:")
    range1 = np.sum((all_indices >= 0) & (all_indices < 40))
    range2 = np.sum((all_indices >= 40) & (all_indices < 80))
    range3 = np.sum((all_indices >= 80) & (all_indices < n_dims))
    print(f"  [0-40]:    {range1:4d} ({100*range1/total:.1f}%) - High RoPE frequency")
    print(f"  [40-80]:   {range2:4d} ({100*range2/total:.1f}%) - Mid RoPE frequency")
    print(f"  [80-128]:  {range3:4d} ({100*range3/total:.1f}%) - Low RoPE frequency (Static outliers)")


def save_static_scales(results, output_path):
    """Save static PRS scales to binary file
    
    Format: k = number of pairs (top_k)
            Each head: 2*k indices + 2*k scales (pair = 2 dimensions)
    """
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('I', MAGIC_SCRS))
        f.write(struct.pack('I', VERSION))
        f.write(struct.pack('I', results['n_layers']))
        f.write(struct.pack('I', results['n_heads']))
        f.write(struct.pack('I', results['n_dims']))
        f.write(struct.pack('I', results['top_k']))  # Number of pairs
        
        # Data: each head stores 2*k indices and 2*k scales
        # Note: extract_static_outliers already pads to 2*top_k, so no need to pad again
        for layer_outliers in results['outliers']:
            for head_outliers in layer_outliers:
                indices = head_outliers['indices']
                scales = head_outliers['scales']
                
                # Verify length (should already be 2*top_k from extract_static_outliers)
                target_len = results['top_k'] * 2
                assert len(indices) == target_len, f"Expected {target_len} indices, got {len(indices)}"
                assert len(scales) == target_len, f"Expected {target_len} scales, got {len(scales)}"
                
                f.write(indices.astype(np.int32).tobytes())
                f.write(scales.astype(np.float32).tobytes())
    
    print(f"\nSaved static PRS scales to: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size} bytes")
    print(f"  Format: k={results['top_k']} pairs, {results['top_k']*2} dimensions per head")


def main():
    parser = argparse.ArgumentParser(description='Extract per-block CRS scales from absmax values')
    parser.add_argument('input_path', help='Path to .bin file with absmax values')
    parser.add_argument('output_path', help='Path to save .bin scales file')
    parser.add_argument('--top-k', type=int, default=2, help='Number of outlier PAIRS per block')
    parser.add_argument('--dim-min', type=int, default=64, help='Minimum dimension index to consider')
    parser.add_argument('--beta', type=float, default=4.0, help='Beta factor for outlier detection (if used)')
    parser.add_argument('--max-scale', type=float, default=None, help='Maximum scale factor (cap) to prevent Wq degradation')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha factor for scale damping (scale = scale^alpha)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed info')
    
    args = parser.parse_args()
    
    data = read_absmax_values(args.input_path)
    if data is None:
        return
        
    results = extract_static_outliers(
        data, 
        top_k=args.top_k, 
        dim_min=args.dim_min, 
        beta=args.beta,
        max_scale=args.max_scale,
        alpha=args.alpha,
        verbose=args.verbose
    )
    
    save_static_scales(results, args.output_path)


if __name__ == '__main__':
    main()
