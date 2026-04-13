#!/usr/bin/env python3
"""
Analyze Outlier-Pair Imbalance for Late Dimensions

Given:
1. scales_k4_target_late.bin: Top-4 outlier pairs per block from dims 64-127
2. rope_absmax_nofuse_wikitrain.bin: Channel-wise absmax values

This script analyzes:
- For each outlier dimension, what is its RoPE pair's absmax?
- Ratio: outlier_absmax / pair_absmax
- If we normalize by outlier_absmax (pair-wise scaling), how much does the pair shrink?
- What fraction of pairs would be quantized to 0 or near-0?

Goal: Show that in late dims, outliers have much larger absmax than their pairs,
      justifying the need for channel-wise (CRS) rather than pair-wise (PRS) scaling.
"""

import struct
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

MAGIC_SCRS = 0x53435253  # "SCRS"
MAGIC_APOR = 0x524F5041  # "APOR"


def read_static_scales(file_path):
    """Read scales_k*.bin file to get outlier indices"""
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_SCRS:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected {hex(MAGIC_SCRS)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        top_k_pairs = struct.unpack('I', f.read(4))[0]
        
        print(f"Static scales file:")
        print(f"  Layers: {n_layers}, Heads: {n_heads}, Dims: {n_dims}")
        print(f"  Top-k pairs: {top_k_pairs} ({top_k_pairs * 2} dims per head)")
        
        outlier_indices = []  # [layer][head] -> list of dim indices
        outlier_scales = []   # [layer][head] -> list of scales
        
        for layer_idx in range(n_layers):
            layer_indices = []
            layer_scales = []
            for head_idx in range(n_heads):
                indices = np.frombuffer(f.read(top_k_pairs * 2 * 4), dtype=np.int32)
                scales = np.frombuffer(f.read(top_k_pairs * 2 * 4), dtype=np.float32)
                
                # Filter out invalid indices (-1)
                valid_mask = indices >= 0
                indices = indices[valid_mask]
                scales = scales[valid_mask]
                
                layer_indices.append(indices)
                layer_scales.append(scales)
            
            outlier_indices.append(layer_indices)
            outlier_scales.append(layer_scales)
        
        return {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'top_k_pairs': top_k_pairs,
            'outlier_indices': outlier_indices,
            'outlier_scales': outlier_scales
        }


def read_absmax_values(file_path):
    """Read rope_absmax_*.bin file"""
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_APOR:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected {hex(MAGIC_APOR)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        
        print(f"\nAbsmax values file:")
        print(f"  Layers: {n_layers}, Heads: {n_heads}, Dims: {n_dims}")
        
        pre_absmax = []   # [layer][head][dim]
        post_absmax = []  # [layer][head][dim]
        
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
            'pre_absmax': pre_absmax,
            'post_absmax': post_absmax
        }


def analyze_outlier_pair_imbalance(scales_data, absmax_data, use_post=True):
    """
    Analyze imbalance within selected top-k RoPE pairs
    
    For each selected pair (even, odd), compute:
    - Ratio: max(even, odd) / min(even, odd)
    - How much the minor channel shrinks under pair-wise scaling
    - Fraction that would quantize to ~0
    
    Args:
        use_post: If True, use post-RoPE absmax; if False, use pre-RoPE absmax
    """
    n_layers = scales_data['n_layers']
    n_heads = scales_data['n_heads']
    
    absmax_source = absmax_data['post_absmax'] if use_post else absmax_data['pre_absmax']
    
    # Accumulators
    all_ratios = []           # major / minor ratios
    all_minor_fractions = []  # minor / major (how much minor shrinks)
    all_major_absmax = []
    all_minor_absmax = []
    
    # For Q4_0 quantization analysis
    # Q4_0: 4-bit, step = 2*scale/15, where scale is typically the absmax
    # If minor_val < 0.5 * step, it quantizes to 0 or ±1
    q4_step_threshold = 0.5  # fraction of step size
    zero_quant_count = 0
    total_count = 0
    
    print(f"\n{'='*60}")
    print(f"Outlier-Pair Imbalance Analysis ({'Post-RoPE' if use_post else 'Pre-RoPE'})")
    print(f"{'='*60}")
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            outlier_indices = scales_data['outlier_indices'][layer_idx][head_idx]
            absmax = absmax_source[layer_idx][head_idx]
            
            # Group indices into pairs (even, odd)
            # outlier_indices contains both even and odd from selected pairs
            # We need to process each pair once
            processed_pairs = set()
            
            for dim_idx in outlier_indices:
                # Get the even index of this pair
                even_idx = (dim_idx // 2) * 2
                odd_idx = even_idx + 1
                
                pair_key = (even_idx, odd_idx)
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                if odd_idx >= len(absmax):
                    continue
                
                even_val = absmax[even_idx]
                odd_val = absmax[odd_idx]
                
                # Major and minor
                major_val = max(even_val, odd_val)
                minor_val = min(even_val, odd_val)
                
                # Calculate ratio
                if minor_val > 1e-10:
                    ratio = major_val / minor_val
                else:
                    ratio = 1000.0  # Cap extreme ratios
                
                all_ratios.append(ratio)
                all_major_absmax.append(major_val)
                all_minor_absmax.append(minor_val)
                
                # Minor fraction (how much minor shrinks when normalized by major)
                minor_fraction = minor_val / major_val if major_val > 1e-10 else 1.0
                all_minor_fractions.append(minor_fraction)
                
                # Q4_0 quantization check
                # If using pair-wise scaling, scale = max(even_val, odd_val) = major_val
                # Step = 2 * major_val / 15
                # Minor quantizes to ~0 if minor_val < threshold * step
                step = 2.0 * major_val / 15.0
                if minor_val < q4_step_threshold * step:
                    zero_quant_count += 1
                total_count += 1
    
    all_ratios = np.array(all_ratios)
    all_minor_fractions = np.array(all_minor_fractions)
    all_major_absmax = np.array(all_major_absmax)
    all_minor_absmax = np.array(all_minor_absmax)
    
    # Print statistics
    print(f"\nTotal pairs analyzed: {total_count:,}")
    print(f"\n{'─'*60}")
    print("RATIO STATISTICS (major / minor):")
    print(f"{'─'*60}")
    print(f"  Mean:     {np.mean(all_ratios):.2f}x")
    print(f"  Median:   {np.median(all_ratios):.2f}x")
    print(f"  Std Dev:  {np.std(all_ratios):.2f}x")
    print(f"  Min:      {np.min(all_ratios):.2f}x")
    print(f"  Max:      {np.max(all_ratios):.2f}x")
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th: {np.percentile(all_ratios, p):.2f}x")
    
    print(f"\n{'─'*60}")
    print("MINOR SHRINKAGE (minor / major):")
    print(f"{'─'*60}")
    print(f"  Mean:     {np.mean(all_minor_fractions):.4f}")
    print(f"  Median:   {np.median(all_minor_fractions):.4f}")
    print(f"  P10:      {np.percentile(all_minor_fractions, 10):.4f}")
    print(f"  P5:       {np.percentile(all_minor_fractions, 5):.4f}")
    print(f"  P1:       {np.percentile(all_minor_fractions, 1):.4f}")
    
    print(f"\n{'─'*60}")
    print("QUANTIZATION IMPACT (Q4_0 simulation):")
    print(f"{'─'*60}")
    zero_quant_rate = zero_quant_count / total_count * 100 if total_count > 0 else 0
    print(f"  Pairs quantized to ~0 under pair-wise scaling: {zero_quant_count:,} / {total_count:,}")
    print(f"  Zero-quantization rate: {zero_quant_rate:.1f}%")
    print(f"  (Threshold: pair < {q4_step_threshold} * step)")
    
    # High imbalance analysis
    high_ratio_thresh = 3.0
    high_ratio_count = np.sum(all_ratios > high_ratio_thresh)
    high_ratio_rate = high_ratio_count / len(all_ratios) * 100
    print(f"\n{'─'*60}")
    print(f"HIGH IMBALANCE (ratio > {high_ratio_thresh}x):")
    print(f"{'─'*60}")
    print(f"  Count: {high_ratio_count:,} / {len(all_ratios):,}")
    print(f"  Rate:  {high_ratio_rate:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"CONCLUSION:")
    print(f"{'='*60}")
    print(f"  {zero_quant_rate:.1f}% of selected pairs have their minor channel")
    print(f"  quantized to ~0 under pair-wise scaling (PRS).")
    print(f"  ")
    print(f"  {high_ratio_rate:.1f}% of pairs show >{high_ratio_thresh}x imbalance,")
    print(f"  indicating severe channel asymmetry in late dimensions.")
    print(f"  ")
    print(f"  -> This justifies channel-wise (CRS) over pair-wise (PRS)")
    print(f"     scaling for late dimensions.")
    print(f"{'='*60}\n")
    
    return {
        'ratios': all_ratios,
        'minor_fractions': all_minor_fractions,
        'major_absmax': all_major_absmax,
        'minor_absmax': all_minor_absmax,
        'zero_quant_count': zero_quant_count,
        'total_count': total_count,
        'high_ratio_count': high_ratio_count
    }


def plot_results(results, output_dir):
    """Generate plots for outlier-pair imbalance analysis"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ratios = results['ratios']
    minor_fractions = results['minor_fractions']
    
    # 1. Histogram of ratios (log scale)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    ax = axes[0]
    ax.hist(np.clip(ratios, 1, 100), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=3.0, color='r', linestyle='--', linewidth=2, label='High imbalance (3x)')
    ax.set_xlabel('Major / Minor Ratio')
    ax.set_ylabel('Count')
    ax.set_title(f'Late Dims Top-4 Pairs: Magnitude Ratio Distribution\n'
                f'Mean: {np.mean(ratios):.2f}x, Median: {np.median(ratios):.2f}x')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. CDF of minor fractions
    ax = axes[1]
    sorted_fractions = np.sort(minor_fractions)
    cdf = np.arange(1, len(sorted_fractions) + 1) / len(sorted_fractions)
    ax.plot(sorted_fractions, cdf, linewidth=2)
    ax.axvline(x=0.1, color='r', linestyle='--', label='10% of major')
    ax.axvline(x=0.2, color='orange', linestyle='--', label='20% of major')
    ax.axvline(x=0.5, color='g', linestyle='--', label='50% of major')
    ax.set_xlabel('Minor / Major Fraction')
    ax.set_ylabel('CDF')
    ax.set_title(f'Minor Channel Shrinkage under Pair-wise Scaling (CDF)\n'
                f'Median: {np.median(minor_fractions):.3f}, P10: {np.percentile(minor_fractions, 10):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plot_path = output_dir / 'outlier_pair_imbalance.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    # 3. Scatter plot: major absmax vs minor absmax
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    major_absmax = results['major_absmax']
    minor_absmax = results['minor_absmax']
    
    # Subsample for visualization if too many points
    if len(major_absmax) > 10000:
        idx = np.random.choice(len(major_absmax), 10000, replace=False)
        major_absmax = major_absmax[idx]
        minor_absmax = minor_absmax[idx]
    
    ax.scatter(major_absmax, minor_absmax, alpha=0.3, s=10)
    
    # Add diagonal line (equal values)
    max_val = max(np.max(major_absmax), np.max(minor_absmax))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Equal (1:1)')
    
    # Add 3x line
    ax.plot([0, max_val], [0, max_val/3], 'orange', linestyle='--', linewidth=1.5, label='3x imbalance')
    
    ax.set_xlabel('Major Channel Absmax')
    ax.set_ylabel('Minor Channel Absmax')
    ax.set_title('Late Dims: Major vs Minor Channel Absmax Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    scatter_path = output_dir / 'outlier_pair_scatter.png'
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    
    print(f"Saved plots to:")
    print(f"  {plot_path}")
    print(f"  {scatter_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze outlier-pair imbalance in late dimensions'
    )
    parser.add_argument('scales_file', help='Path to scales_k*.bin file')
    parser.add_argument('absmax_file', help='Path to rope_absmax_*.bin file')
    parser.add_argument('--output-dir', default='outlier_pair_analysis',
                       help='Output directory for plots and results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--use-pre', action='store_true',
                       help='Use pre-RoPE absmax instead of post-RoPE')
    
    args = parser.parse_args()
    
    # Read data
    scales_data = read_static_scales(args.scales_file)
    absmax_data = read_absmax_values(args.absmax_file)
    
    # Analyze
    results = analyze_outlier_pair_imbalance(
        scales_data, absmax_data, use_post=not args.use_pre
    )
    
    # Plot
    if args.plot:
        plot_results(results, args.output_dir)


if __name__ == '__main__':
    main()
