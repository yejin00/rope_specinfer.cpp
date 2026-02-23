#!/usr/bin/env python3
"""
Extract Static CRS Scales from rope_absmax_values.bin

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


def extract_static_outliers(data, top_k=8, dim_min=80, beta=1.0, verbose=True):
    """
    Extract static outlier channels from Pre-RoPE absmax values.
    
    Static outliers are channels that consistently have large absmax BEFORE RoPE.
    Only considers channels >= dim_min (default 80) for static outliers.
    
    Args:
        beta: Scale factor adjustment. scale = absmax / beta
    """
    n_layers = data['n_layers']
    n_heads = data['n_heads']
    n_dims = data['n_dims']
    
    results = {
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_dims': n_dims,
        'top_k': top_k,
        'dim_min': dim_min,
        'beta': beta,
        'outliers': []
    }
    
    print(f"\nExtracting static outliers with PRS (top-{top_k} pairs per head, dim >= {dim_min}, β={beta:.2f})...")
    
    for layer_idx in range(n_layers):
        layer_outliers = []
        
        for head_idx in range(n_heads):
            # Pre-RoPE channel-wise absmax
            pre_absmax = data['pre_absmax'][layer_idx][head_idx]
            
            # Find top-k channels by absmax (from ALL dims)
            top_indices = np.argsort(pre_absmax)[-top_k:][::-1]
            top_scales = pre_absmax[top_indices]
            
            # Filter: keep only channels >= dim_min for static PRS
            mask = top_indices >= dim_min
            filtered_indices = top_indices[mask]
            filtered_scales = top_scales[mask]
            
            # Convert to RoPE pairs for PRS (Pair-wise Rotary Scaling)
            # RoPE operates on pairs (d_even, d_odd) where d_even = d & ~1, d_odd = d | 1
            pair_set = set()
            pair_scales = {}
            
            for idx, scale in zip(filtered_indices, filtered_scales):
                # Get RoPE pair
                d_even = int(idx & ~1)  # Even dimension
                d_odd = int(idx | 1)     # Odd dimension
                
                # Add pair (use max scale from both dimensions)
                pair_key = (d_even, d_odd)
                if pair_key not in pair_set:
                    pair_set.add(pair_key)
                    # Use max absmax from the pair for conservative scaling
                    pair_scale = max(pre_absmax[d_even], pre_absmax[d_odd])
                    pair_scales[pair_key] = pair_scale
            
            # Calculate median of non-PRS channels for beta normalization
            # Exclude ALL dimensions in PRS pairs (both outliers and their pairs)
            prs_dims = set()
            for (d_even, d_odd) in pair_set:
                prs_dims.add(d_even)
                prs_dims.add(d_odd)
            
            # Get median of remaining channels (ALL dims, not in PRS pairs)
            non_prs_mask = np.ones(len(pre_absmax), dtype=bool)
            for d in prs_dims:
                if d < len(pre_absmax):
                    non_prs_mask[d] = False
            
            if np.sum(non_prs_mask) > 0:
                median_normal = np.median(pre_absmax[non_prs_mask])
            else:
                median_normal = 1.0  # Fallback
            
            # Flatten pairs back to individual dimensions
            prs_indices = []
            prs_scales = []
            for (d_even, d_odd), absmax in sorted(pair_scales.items()):
                prs_indices.extend([d_even, d_odd])
                # Scale to normalize outlier to median level
                # scale = outlier_absmax / median → K_new = K_orig / scale = median
                final_scale = absmax / median_normal if median_normal > 0 else 1.0
                prs_scales.extend([final_scale, final_scale])  # Same scale for both in pair
            
            prs_indices = np.array(prs_indices, dtype=np.int32)
            prs_scales = np.array(prs_scales, dtype=np.float32)
            
            # Limit to top_k pairs (= 2*top_k dimensions)
            if len(prs_indices) > top_k * 2:
                prs_indices = prs_indices[:top_k * 2]
                prs_scales = prs_scales[:top_k * 2]
            
            # Pad to top_k pairs with -1 (invalid)
            target_len = top_k * 2
            if len(prs_indices) < target_len:
                pad_len = target_len - len(prs_indices)
                prs_indices = np.concatenate([prs_indices, np.full(pad_len, -1, dtype=np.int32)])
                prs_scales = np.concatenate([prs_scales, np.ones(pad_len, dtype=np.float32)])
            
            layer_outliers.append({
                'indices': prs_indices.astype(np.int32),
                'scales': prs_scales.astype(np.float32)
            })
            
            if verbose and layer_idx % 8 == 0 and head_idx == 0:
                print(f"  Layer {layer_idx}, Head {head_idx}:")
                print(f"    Top-{top_k} outlier channels: {top_indices}")
                print(f"    Scales (absmax): {np.round(top_scales, 2)}")
                in_range = np.sum(top_indices >= dim_min)
                print(f"    Channels in static range (>={dim_min}): {in_range}/{top_k}")
                # Show PRS pairs
                valid_pairs = [(i, s) for i, s in zip(prs_indices, prs_scales) if i >= 0]
                if valid_pairs:
                    pairs_str = ", ".join([f"({valid_pairs[i][0]},{valid_pairs[i+1][0]})" 
                                          for i in range(0, len(valid_pairs), 2)])
                    print(f"    PRS pairs: {pairs_str}")
                    # Show median and beta info
                    print(f"    Median (non-PRS, all dims): {median_normal:.4f}")
                    if len(pair_scales) > 0:
                        avg_outlier = np.mean(list(pair_scales.values()))
                        effective_beta = avg_outlier / median_normal if median_normal > 0 else 1.0
                        print(f"    Avg outlier absmax: {avg_outlier:.4f}")
                        print(f"    Effective β (outlier/median): {effective_beta:.2f}")
                        print(f"    → Scales use per-head median (normalize outlier → {median_normal:.2f})")

                        pairs_detail = []
                        for (d_even, d_odd), absmax in sorted(pair_scales.items()):
                            scale_val = absmax / median_normal if median_normal > 0 else 1.0
                            pairs_detail.append(f"({d_even},{d_odd}): absmax={absmax:.4f}, scale={scale_val:.4f}")
                        if pairs_detail:
                            print(f"    Pair details: " + "; ".join(pairs_detail))
        
        results['outliers'].append(layer_outliers)
    
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
    parser = argparse.ArgumentParser(description='Extract static CRS scales from absmax values')
    parser.add_argument('absmax_file', help='Path to rope_absmax_values.bin')
    parser.add_argument('output_file', nargs='?', help='Output path (scales_static_k.bin)')
    parser.add_argument('--top-k', type=int, default=8, help='Top outlier channels per head (default: 8)')
    parser.add_argument('--dim-min', type=int, default=80, help='Minimum dimension for static outliers (default: 80)')
    parser.add_argument('--beta', type=float, default=1.0, help='Scale factor: scale = absmax / beta (default: 1.0)')
    parser.add_argument('--analyze-beta', action='store_true', help='Analyze and suggest optimal beta, then exit')
    parser.add_argument('--verbose', action='store_true', help='Print detailed info')
    
    args = parser.parse_args()
    
    print(f"Reading {args.absmax_file}...")
    data = read_absmax_values(args.absmax_file)
    
    # Analyze beta mode
    if args.analyze_beta:
        suggested_beta = analyze_beta(data, top_k=args.top_k, dim_min=args.dim_min)
        print(f"Run with: --beta {suggested_beta:.2f}")
        return
    
    if not args.output_file:
        print("Error: output_file required (unless using --analyze-beta)")
        return
    
    results = extract_static_outliers(data, top_k=args.top_k, dim_min=args.dim_min, beta=args.beta, verbose=args.verbose)
    
    analyze_distribution(data, results)
    
    save_static_scales(results, args.output_file)
    
    print(f"\nDone! Use with: --crs-scales-k {args.output_file}")


if __name__ == '__main__':
    main()
