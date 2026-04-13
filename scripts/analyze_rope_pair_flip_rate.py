#!/usr/bin/env python3
"""
Analyze RoPE Pair Flip Rate in Activations

This script calculates the "pair flip rate" for each RoPE pair (even, odd) dimension:
- For each token, for each pair (dim_even, dim_odd), check if |even| > |odd|
- Calculate pe = P(|even| > |odd|) for each dimension pair
- If pe is close to 0.5, the pair flips frequently (unstable dominance)
- If pe is close to 0 or 1, the pair has stable dominance

Early dimensions (0-63) are expected to have pe closer to 0.5 due to
larger rotation angles, while late dimensions (64-127) should have pe
closer to 0 or 1 (more stable).

Input file format (ROPV - post-RoPE values):
- Magic: 0x524F5056 ("ROPV")
- Version: uint32
- n_layers: uint32
- n_heads: uint32
- n_dims: uint32
- n_tokens: uint32
- For each layer:
  - pre_count: uint32
  - pre_values[pre_count]: float32 (Pre-RoPE values, flattened)
  - post_count: uint32
  - post_values[post_count]: float32 (Post-RoPE values, flattened)

The script will analyze the post-RoPE values to measure pair flip rates.
"""

import os
import sys
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path

MAGIC_ROPV = 0x524F5056  # "ROPV"

def read_ropv_header(f):
    """Read ROPV file header"""
    magic = struct.unpack('I', f.read(4))[0]
    if magic != MAGIC_ROPV:
        raise ValueError(f"Invalid magic: 0x{magic:08X}, expected: 0x{MAGIC_ROPV:08X}")
    
    version = struct.unpack('I', f.read(4))[0]
    n_layers = struct.unpack('I', f.read(4))[0]
    n_heads = struct.unpack('I', f.read(4))[0]
    n_dims = struct.unpack('I', f.read(4))[0]
    n_tokens = struct.unpack('I', f.read(4))[0]
    
    return version, n_layers, n_heads, n_dims, n_tokens

def analyze_pair_flip_rate(file_path, target_layers=None, target_heads=None, output_dir=None, plot=False):
    """
    Analyze pair flip rate in post-RoPE activations
    
    Args:
        file_path: Path to ROPV file
        target_layers: List of layer indices to analyze (None = all)
        target_heads: List of head indices to analyze (None = all)
        output_dir: Directory to save outputs (None = no save)
        plot: Whether to generate and save plots
    """
    with open(file_path, 'rb') as f:
        version, n_layers, n_heads, n_dims, n_tokens = read_ropv_header(f)
        
        print(f"ROPV file: {file_path}")
        print(f"Version: {version}")
        print(f"Layers: {n_layers}, Heads: {n_heads}, Dims: {n_dims}, Tokens: {n_tokens}")
        
        if target_layers is None:
            target_layers = list(range(n_layers))
        if target_heads is None:
            target_heads = list(range(n_heads))
            
        print(f"Analyzing layers: {target_layers}")
        print(f"Analyzing heads: {target_heads}")
        
        # Results will be stored here
        results = {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'n_tokens_analyzed': 0,
            'layers': {}
        }
        
        # Process each layer
        for layer_idx in range(n_layers):
            # In these specific files, the data is stored in the 'pre_count' field 
            # even though it represents post-RoPE values.
            pre_count = struct.unpack('I', f.read(4))[0]
            
            if layer_idx in target_layers:
                print(f"Processing layer {layer_idx}...")
                
                # Read the actual values (stored in pre_count field)
                if pre_count > 0:
                    values = np.frombuffer(f.read(pre_count * 4), dtype=np.float32)
                    n_tokens_layer = pre_count // (n_heads * n_dims)
                    print(f"  Read {n_tokens_layer} tokens")
                else:
                    values = np.array([])
                    n_tokens_layer = 0
                    print(f"  No data for layer {layer_idx}")
                
                # Still need to read post_count to advance the file pointer properly
                post_count = struct.unpack('I', f.read(4))[0]
                if post_count > 0:
                    f.seek(post_count * 4, 1)
                
                if n_tokens_layer == 0:
                    continue
                    
                results['n_tokens_analyzed'] = max(results['n_tokens_analyzed'], n_tokens_layer)
                
                # Reshape to [n_tokens, n_heads, n_dims]
                values = values.reshape(n_tokens_layer, n_heads, n_dims)
                
                layer_results = {}
                results['layers'][layer_idx] = layer_results
                
                # Process each target head
                for head_idx in target_heads:
                    head_results = {}
                    layer_results[head_idx] = head_results
                    
                    # Process each dimension pair
                    head_values = np.abs(values[:, head_idx, :])  # [n_tokens, n_dims]
                    
                    # Dictionary to store pair flip rates for different dimension ranges
                    pair_flip_rates = {
                        'all': [],          # All pairs (0-127)
                        'early': [],        # Early pairs (0-63)
                        'late': [],         # Late pairs (64-127)
                        'very_early': [],   # Very early pairs (0-31)
                        'mid_early': [],    # Mid early pairs (32-63)
                        'mid_late': [],     # Mid late pairs (64-95)
                        'very_late': []     # Very late pairs (96-127)
                    }
                    
                    # Analyze each RoPE pair (even-odd pairs)
                    for d in range(0, n_dims, 2):
                        if d + 1 >= n_dims:
                            continue  # Skip last odd dimension if n_dims is odd
                        
                        even_vals = head_values[:, d]      # [n_tokens]
                        odd_vals = head_values[:, d + 1]   # [n_tokens]
                        
                        # Count how many tokens have |even| > |odd|
                        even_dominates = np.sum(even_vals > odd_vals)
                        
                        # Calculate probability P(|even| > |odd|)
                        p_even = even_dominates / n_tokens_layer
                        
                        # Calculate flip rate (how often dominance changes)
                        # If p_even is close to 0 or 1, one dimension always dominates (stable)
                        # If p_even is close to 0.5, the dominance flips frequently (unstable)
                        flip_rate = 2 * min(p_even, 1 - p_even)  # Will be 1.0 when p_even=0.5, and 0.0 when p_even=0 or 1
                        
                        pair_idx = d // 2
                        pair_data = {
                            'pair_idx': pair_idx,
                            'dim_even': d,
                            'dim_odd': d + 1,
                            'p_even': float(p_even),
                            'p_odd': float(1.0 - p_even),
                            'flip_rate': float(flip_rate)
                        }
                        
                        # Store in appropriate dimension range
                        pair_flip_rates['all'].append(pair_data)
                        
                        if d < 64:
                            pair_flip_rates['early'].append(pair_data)
                            if d < 32:
                                pair_flip_rates['very_early'].append(pair_data)
                            else:
                                pair_flip_rates['mid_early'].append(pair_data)
                        else:
                            pair_flip_rates['late'].append(pair_data)
                            if d < 96:
                                pair_flip_rates['mid_late'].append(pair_data)
                            else:
                                pair_flip_rates['very_late'].append(pair_data)
                    
                    # Store results for this head
                    head_results['pair_flip_rates'] = pair_flip_rates
                    
                    # Calculate summary statistics
                    summary_stats = {}
                    for range_name, pairs in pair_flip_rates.items():
                        flip_rates = [p['flip_rate'] for p in pairs]
                        if flip_rates:
                            summary_stats[range_name] = {
                                'mean_flip_rate': float(np.mean(flip_rates)),
                                'median_flip_rate': float(np.median(flip_rates)),
                                'min_flip_rate': float(np.min(flip_rates)),
                                'max_flip_rate': float(np.max(flip_rates)),
                                'std_flip_rate': float(np.std(flip_rates)),
                                'n_pairs': len(flip_rates)
                            }
                    
                    head_results['summary'] = summary_stats
            else:
                # Skip values for non-target layers
                if pre_count > 0:
                    f.seek(pre_count * 4, 1)
                post_count = struct.unpack('I', f.read(4))[0]
                if post_count > 0:
                    f.seek(post_count * 4, 1)
    
    # Print summary results
    print("\n=== Pair Flip Rate Summary ===")
    
    # Initialize global statistics accumulators
    global_stats = {
        'all': [],
        'early': [],
        'late': [],
        'very_early': [],
        'mid_early': [],
        'mid_late': [],
        'very_late': []
    }
    
    for layer_idx in results['layers']:
        for head_idx in results['layers'][layer_idx]:
            head_data = results['layers'][layer_idx][head_idx]
            summary = head_data['summary']
            
            # Accumulate individual pair flip rates for global stats
            for range_name, pairs in head_data['pair_flip_rates'].items():
                flip_rates = [p['flip_rate'] for p in pairs]
                global_stats[range_name].extend(flip_rates)
                
            # Accumulate per-pair-index stats for global plot
            if 'pair_idx_stats' not in results:
                results['pair_idx_stats'] = {}
                
            for pair in head_data['pair_flip_rates']['all']:
                idx = pair['pair_idx']
                if idx not in results['pair_idx_stats']:
                    results['pair_idx_stats'][idx] = {'flip_rates': [], 'p_evens': []}
                results['pair_idx_stats'][idx]['flip_rates'].append(pair['flip_rate'])
                results['pair_idx_stats'][idx]['p_evens'].append(pair['p_even'])
            
            if len(results['layers']) <= 4 and len(target_heads) <= 2:
                # Only print detailed head stats if we're not running on everything
                print(f"\nLayer {layer_idx}, Head {head_idx}:")
                
                for range_name, stats in summary.items():
                    print(f"  {range_name.upper()} pairs ({stats['n_pairs']} pairs):")
                    print(f"    Mean flip rate: {stats['mean_flip_rate']:.4f}")
                    print(f"    Median flip rate: {stats['median_flip_rate']:.4f}")
                    print(f"    Range: [{stats['min_flip_rate']:.4f}, {stats['max_flip_rate']:.4f}]")
                
                # Compute early vs late ratio
                if 'early' in summary and 'late' in summary:
                    early_rate = summary['early']['mean_flip_rate']
                    late_rate = summary['late']['mean_flip_rate']
                    ratio = early_rate / late_rate if late_rate > 0 else float('inf')
                    print(f"  Early/Late ratio: {ratio:.2f}x")
    
    # Print Global Statistics
    print("\n" + "="*40)
    print("GLOBAL STATISTICS (All Analyzed Layers/Heads)")
    print("="*40)
    for range_name in ['all', 'early', 'late', 'very_early', 'mid_early', 'mid_late', 'very_late']:
        rates = global_stats[range_name]
        if rates:
            print(f"  {range_name.upper()} pairs (Total {len(rates)} pairs):")
            print(f"    Mean flip rate: {np.mean(rates):.4f}")
            print(f"    Median flip rate: {np.median(rates):.4f}")
            print(f"    Std dev: {np.std(rates):.4f}")
            
    if global_stats['early'] and global_stats['late']:
        early_mean = np.mean(global_stats['early'])
        late_mean = np.mean(global_stats['late'])
        ratio = early_mean / late_mean if late_mean > 0 else float('inf')
        print(f"\n  GLOBAL Early/Late Flip Rate Ratio: {ratio:.2f}x")
        print("  (Higher ratio confirms Early dims flip more frequently than Late dims)")
    print("="*40 + "\n")
    
    # Save results to JSON if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = output_dir / f"pair_flip_rates_{Path(file_path).stem}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved detailed results to: {json_path}")
    
    # Generate and save plots if requested
    if plot and output_dir:
        plot_flip_rates(results, output_dir)
    
    return results

def plot_flip_rates(results, output_dir):
    """Generate and save plots for pair flip rates"""
    output_dir = Path(output_dir)
    
    # 1. Generate Global Average Plot
    if 'pair_idx_stats' in results:
        pair_indices = sorted(list(results['pair_idx_stats'].keys()))
        avg_flip_rates = [np.mean(results['pair_idx_stats'][idx]['flip_rates']) for idx in pair_indices]
        avg_p_evens = [np.mean(results['pair_idx_stats'][idx]['p_evens']) for idx in pair_indices]
        std_flip_rates = [np.std(results['pair_idx_stats'][idx]['flip_rates']) for idx in pair_indices]
        
        plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 1, height_ratios=[2, 1])
        
        # Plot Global Average Flip Rates
        ax1 = plt.subplot(gs[0])
        ax1.plot(pair_indices, avg_flip_rates, 'o-', linewidth=2, label='Mean Flip Rate')
        ax1.fill_between(pair_indices, 
                         np.maximum(0, np.array(avg_flip_rates) - np.array(std_flip_rates)),
                         np.minimum(1, np.array(avg_flip_rates) + np.array(std_flip_rates)),
                         alpha=0.2, label='±1 Std Dev')
        ax1.axhline(y=0.5, color='r', linestyle='--', label='Max Flip Rate (0.5)')
        ax1.axvline(x=16, color='g', linestyle='--', label='Dim 32 (Block 1)')
        ax1.axvline(x=32, color='m', linestyle='--', label='Dim 64 (Block 2)')
        ax1.axvline(x=48, color='c', linestyle='--', label='Dim 96 (Block 3)')
        ax1.set_xlabel('Dimension Pair Index')
        ax1.set_ylabel('Average Pair Flip Rate (Across all layers/heads)')
        ax1.set_title(f'GLOBAL AVERAGE: RoPE Pair Flip Rates (N={len(results["layers"])} layers, {len(results["layers"][list(results["layers"].keys())[0]])} heads)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot Global Average P(|even| > |odd|)
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(pair_indices, avg_p_evens, 'o-', color='purple', linewidth=2, label='Mean P(|even| > |odd|)')
        ax2.axhline(y=0.5, color='r', linestyle='--', label='P=0.5')
        ax2.axvline(x=16, color='g', linestyle='--')
        ax2.axvline(x=32, color='m', linestyle='--')
        ax2.axvline(x=48, color='c', linestyle='--')
        ax2.set_xlabel('Dimension Pair Index')
        ax2.set_ylabel('Mean P(|even| > |odd|)')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = output_dir / "global_average_pair_flip_rates.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved global average plot to: {plot_path}")
        
    # 2. Optionally, keep generating individual plots if there are only a few
    num_heads_analyzed = sum(len(h) for h in results['layers'].values())
    if num_heads_analyzed <= 4:  # Only plot individual if 4 or fewer to avoid spam
        for layer_idx in results['layers']:
            for head_idx in results['layers'][layer_idx]:
                head_data = results['layers'][layer_idx][head_idx]
                pairs_all = head_data['pair_flip_rates']['all']
            
            if not pairs_all:
                continue
                
            # Sort by pair_idx for plotting
            pairs_all.sort(key=lambda x: x['pair_idx'])
            pair_indices = [p['pair_idx'] for p in pairs_all]
            flip_rates = [p['flip_rate'] for p in pairs_all]
            p_even = [p['p_even'] for p in pairs_all]
            
            # Plot flip rates
            plt.figure(figsize=(12, 8))
            gs = GridSpec(2, 1, height_ratios=[2, 1])
            
            # Plot flip rates
            ax1 = plt.subplot(gs[0])
            ax1.plot(pair_indices, flip_rates, 'o-', label='Flip Rate')
            ax1.axhline(y=0.5, color='r', linestyle='--', label='Max Flip Rate (0.5)')
            ax1.axvline(x=16, color='g', linestyle='--', label='Dim 32 (Block 1)')
            ax1.axvline(x=32, color='m', linestyle='--', label='Dim 64 (Block 2)')
            ax1.axvline(x=48, color='c', linestyle='--', label='Dim 96 (Block 3)')
            ax1.set_xlabel('Dimension Pair Index')
            ax1.set_ylabel('Pair Flip Rate (2 * min(p_even, p_odd))')
            ax1.set_title(f'Layer {layer_idx}, Head {head_idx}: Pair Flip Rates')
            ax1.grid(True)
            ax1.legend()
            
            # Plot p_even probabilities
            ax2 = plt.subplot(gs[1], sharex=ax1)
            ax2.plot(pair_indices, p_even, 'o-', color='purple', label='P(|even| > |odd|)')
            ax2.axhline(y=0.5, color='r', linestyle='--', label='P=0.5')
            ax2.axvline(x=16, color='g', linestyle='--')
            ax2.axvline(x=32, color='m', linestyle='--')
            ax2.axvline(x=48, color='c', linestyle='--')
            ax2.set_xlabel('Dimension Pair Index')
            ax2.set_ylabel('P(|even| > |odd|)')
            ax2.set_ylim(0, 1)
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            
            plot_path = output_dir / f"pair_flip_rates_l{layer_idx}_h{head_idx}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            
            # Create histograms of flip rates for early vs late dimensions
            plt.figure(figsize=(12, 6))
            
            # Early dimensions (0-63)
            early_pairs = head_data['pair_flip_rates']['early']
            early_rates = [p['flip_rate'] for p in early_pairs] if early_pairs else []
            
            # Late dimensions (64-127)
            late_pairs = head_data['pair_flip_rates']['late']
            late_rates = [p['flip_rate'] for p in late_pairs] if late_pairs else []
            
            plt.hist(early_rates, bins=20, alpha=0.7, label=f'Early (0-63): mean={np.mean(early_rates):.3f}')
            plt.hist(late_rates, bins=20, alpha=0.7, label=f'Late (64-127): mean={np.mean(late_rates):.3f}')
            plt.xlabel('Pair Flip Rate')
            plt.ylabel('Count')
            plt.title(f'Layer {layer_idx}, Head {head_idx}: Distribution of Flip Rates')
            plt.legend()
            plt.grid(True)
            
            hist_path = output_dir / f"pair_flip_hist_l{layer_idx}_h{head_idx}.png"
            plt.savefig(hist_path, dpi=150)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze RoPE pair flip rates in activations')
    parser.add_argument('file_path', help='Path to activation dump file (ROPV format)')
    parser.add_argument('--layers', type=str, default=None, 
                        help='Comma-separated list of layer indices to analyze (default: all)')
    parser.add_argument('--heads', type=str, default=None,
                        help='Comma-separated list of head indices to analyze (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save output files')
    parser.add_argument('--plot', action='store_true',
                        help='Generate and save plots')
    
    args = parser.parse_args()
    
    target_layers = [int(x) for x in args.layers.split(',')] if args.layers else None
    target_heads = [int(x) for x in args.heads.split(',')] if args.heads else None
    
    analyze_pair_flip_rate(args.file_path, target_layers, target_heads, args.output_dir, args.plot)

if __name__ == '__main__':
    main()
