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


def extract_static_outliers(data, percentile_cutoff=20, dim_min=0, dim_end=64, beta=4.0, max_scale=None, alpha=1.0, late_target=False, verbose=True):
    """
    Extract static scales for ALL early dimension pairs (except bottom percentile).
    
    Strategy:
    1. Filter pairs by L2 norm: exclude bottom percentile_cutoff% (default 20%)
    2. Apply scaling to remaining pairs

    Target mode (--late-target):
    - False (default): target = median(sqrt_max of early pairs)
    - True: target = median(absmax of late dims, i.e. interleaved dim_end..n_dims)
            Uses raw pair_max (not sqrt_max) to match units with raw late_median.
            Brings early dims DOWN to the same level as late dims -> more uniform
            per-head quantization distribution.
    
    Args:
        percentile_cutoff: Exclude pairs in bottom X% by L2 norm (default 20)
        dim_min: Start dimension index (default 0)
        dim_end: End dimension index exclusive (default 64)
        alpha: Scaling strength multiplier (default 1.0)
        max_scale: Maximum allowed scale factor
        late_target: If True, use median of late dims as cross-band target
        verbose: Print detailed info
    """
    n_layers = data['n_layers']
    n_heads = data['n_heads']
    n_dims = data['n_dims']
    
    # Calculate max possible pairs in range
    num_pairs_in_range = (dim_end - dim_min) // 2
    max_outliers = num_pairs_in_range * 2  # Store all pairs (2 dims each)
    
    results = {
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_dims': n_dims,
        'top_k': num_pairs_in_range,  # All pairs in range
        'dim_min': dim_min,
        'dim_end': dim_end,
        'percentile_cutoff': percentile_cutoff,
        'alpha': alpha,
        'outliers': []
    }
    
    # Statistics tracking
    all_scales_raw = []
    total_pairs = 0
    filtered_pairs = 0

    print(f"\n=== All-Pair Scaling (Early Dims, L2 Filter + Scaling) ===")
    print(f"Strategy: Scale ALL pairs in range [{dim_min}, {dim_end}), exclude bottom {percentile_cutoff}%")
    print(f"  1. Filter by L2 norm: exclude bottom {percentile_cutoff}% pairs")
    if late_target:
        print(f"  2. Cross-band target: median of LATE dims (interleaved {dim_end}..{data['n_dims']}) [raw pair_max]")
    else:
        print(f"  2. Target: median of early pair sqrt_max values")
    print(f"Total pairs per head: {num_pairs_in_range}")
    if alpha != 1.0:
        print(f"Alpha (Scaling strength): {alpha:.2f}")
    if max_scale:
        print(f"Max Scale Limit: {max_scale:.2f}")
    
    for layer_idx in range(n_layers):
        layer_outliers = []
        
        for head_idx in range(n_heads):
            original_absmax = data['pre_absmax'][layer_idx][head_idx]
            
            # Get absmax values for early dimensions only
            early_absmax = original_absmax[dim_min:dim_end]
            
            # Step 1: Calculate L2 norm for each pair
            pair_l2_norms = []
            for i in range(0, len(early_absmax), 2):
                if i + 1 < len(early_absmax):
                    d_even = dim_min + i
                    d_odd = dim_min + i + 1
                    # L2 norm: sqrt(a^2 + b^2)
                    l2_norm = np.sqrt(early_absmax[i]**2 + early_absmax[i+1]**2)
                    sqrt_max = np.sqrt(max(early_absmax[i], early_absmax[i+1]))
                    pair_l2_norms.append((l2_norm, sqrt_max, d_even, d_odd))
            
            total_pairs += len(pair_l2_norms)
            
            # Step 2: Filter out bottom percentile_cutoff% by L2 norm
            if len(pair_l2_norms) > 0:
                l2_values = np.array([p[0] for p in pair_l2_norms])
                threshold = np.percentile(l2_values, percentile_cutoff)
                selected_pairs = [p for p in pair_l2_norms if p[0] >= threshold]
                filtered_pairs += (len(pair_l2_norms) - len(selected_pairs))
            else:
                selected_pairs = []
            
            # Step 3: Calculate target value
            if late_target:
                # Cross-band: use median of raw late dim absmax as target
                late_absmax = original_absmax[dim_end:]
                target_val = float(np.median(late_absmax)) if len(late_absmax) > 0 else 1.0
            else:
                # Default: median of early pair sqrt_max
                if len(selected_pairs) > 0:
                    sqrt_max_values = [p[1] for p in selected_pairs]
                    target_val = np.median(sqrt_max_values)
                else:
                    target_val = 1.0

            if target_val < 1e-6:
                target_val = 1.0

            # Step 4: Apply scaling to selected pairs
            prs_indices = []
            prs_scales = []

            for l2_norm, sqrt_max, d_even, d_odd in selected_pairs:
                prs_indices.extend([d_even, d_odd])

                if late_target:
                    # Use raw pair_max (unit consistent with raw late_median)
                    pair_max = sqrt_max ** 2  # sqrt_max = sqrt(max(a,b)) -> pair_max = max(a,b)
                    raw_scale = (pair_max / target_val) ** alpha if target_val > 1e-6 else 1.0
                else:
                    # Default: sqrt_max vs early median
                    raw_scale = (sqrt_max / target_val) ** alpha if target_val > 1e-6 else 1.0
                final_scale = max(1.0, min(raw_scale, max_scale if max_scale else float('inf')))
                
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
                valid_count = len(selected_pairs)
                target_label = "late_median" if late_target else "early_sqrt_max_median"
                print(f"  Layer {layer_idx}, Head {head_idx}:")
                print(f"    Total pairs: {len(pair_l2_norms)}, Selected: {valid_count} ({100*valid_count/len(pair_l2_norms):.1f}%)")
                print(f"    L2 threshold: {threshold:.4f}, Target [{target_label}]: {target_val:.4f}")
        
        results['outliers'].append(layer_outliers)
    
    # Print Scale Stats
    print(f"\n[Filtering Statistics]")
    print(f"  Total pairs: {total_pairs}")
    print(f"  Filtered out (bottom {percentile_cutoff}%): {filtered_pairs} ({100*filtered_pairs/total_pairs:.1f}%)")
    print(f"  Scaled pairs: {total_pairs - filtered_pairs} ({100*(total_pairs-filtered_pairs)/total_pairs:.1f}%)")
    
    if len(all_scales_raw) > 0:
        all_scales_raw = np.array(all_scales_raw)
        print(f"\n[Scale Statistics]")
        print(f"  Count: {len(all_scales_raw)}")
        print(f"  Min: {all_scales_raw.min():.4f}")
        print(f"  Max: {all_scales_raw.max():.4f}")
        print(f"  Mean: {all_scales_raw.mean():.4f}")
        print(f"  Median: {np.median(all_scales_raw):.4f}")
        if max_scale:
            n_clipped = np.sum(all_scales_raw >= max_scale)
            print(f"  Clipped: {n_clipped} ({n_clipped/len(all_scales_raw)*100:.1f}%) >= {max_scale}")

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
    
    print(f"\nSaved static PRS scales (early dims, per-head, pair-wise) to: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size} bytes")
    print(f"  Format: k={results['top_k']} pairs, {results['top_k']*2} dimensions per head")


def main():
    parser = argparse.ArgumentParser(description='Extract per-head CRS scales from absmax values (ALL early pairs, L2 filter + sqrt_max scale)')
    parser.add_argument('input_path', help='Path to .bin file with absmax values')
    parser.add_argument('output_path', help='Path to save .bin scales file')
    parser.add_argument('--percentile-cutoff', type=float, default=20.0, help='Exclude bottom X%% pairs by L2 norm (default 20)')
    parser.add_argument('--dim-min', type=int, default=0, help='Minimum dimension index to consider')
    parser.add_argument('--dim-end', type=int, default=64, help='End dimension index (exclusive) to consider')
    parser.add_argument('--max-scale', type=float, default=None, help='Maximum scale factor (cap) to prevent Wq degradation')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha factor for scaling strength')
    parser.add_argument('--late-target', action='store_true', help='Use median of late dims (interleaved dim_end..n_dims) as target instead of early pair median')
    parser.add_argument('--verbose', action='store_true', help='Print detailed info')
    
    args = parser.parse_args()
    
    data = read_absmax_values(args.input_path)
    if data is None:
        return
        
    results = extract_static_outliers(
        data, 
        percentile_cutoff=args.percentile_cutoff, 
        dim_min=args.dim_min, 
        dim_end=args.dim_end,
        max_scale=args.max_scale,
        alpha=args.alpha,
        late_target=args.late_target,
        verbose=args.verbose
    )
    
    save_static_scales(results, args.output_path)


if __name__ == '__main__':
    main()
