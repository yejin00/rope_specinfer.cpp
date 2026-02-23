#!/usr/bin/env python3
"""
Analyze PRS Matching Rate:
- Load statically selected dims from PRS scales
- Compare with actual runtime block absmax argmax dims
- Calculate matching rate to validate PRS effectiveness
"""

import struct
import numpy as np
import json
import argparse
from collections import defaultdict

def read_prs_scales(scales_file):
    """
    Read PRS scales binary file and extract selected dims.
    
    Format:
    - magic (4 bytes): 'SRCS'
    - version (4 bytes)
    - n_layers (4 bytes)
    - n_heads (4 bytes)
    - n_dims (4 bytes)
    - top_k (4 bytes): number of pairs
    
    For each layer, head:
    - indices[2*top_k] (int32)
    - scales[2*top_k] (float32)
    """
    selected_dims = {}  # key: (layer, head, block) -> set of dims
    
    with open(scales_file, 'rb') as f:
        magic = f.read(4)
        if magic != b'SRCS':  # Static Row-wise Channel Scaling
            raise ValueError(f"Invalid magic: {magic}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        top_k = struct.unpack('I', f.read(4))[0]
        
        n_outliers = top_k * 2  # 2 dims per pair
        
        print(f"PRS Scales: layers={n_layers}, heads={n_heads}, dims={n_dims}, top_k={top_k} (pairs)")
        print(f"  Each head has {n_outliers} selected dims")
        
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                # Read fixed-length arrays
                indices = np.frombuffer(f.read(n_outliers * 4), dtype=np.int32)
                scales = np.frombuffer(f.read(n_outliers * 4), dtype=np.float32)
                
                # Group dims by block
                # JSON uses relative block indices from dim_min=64:
                #   Block 0 = dims [64, 96]
                #   Block 1 = dims [96, 128]
                block_dims = defaultdict(set)
                for dim in indices:
                    if dim >= 64:  # Only dims 64+
                        # Convert absolute dim to JSON block index
                        if 64 <= dim < 96:
                            json_block_idx = 0
                        elif 96 <= dim < 128:
                            json_block_idx = 1
                        else:
                            continue
                        block_dims[json_block_idx].add(int(dim))
                
                for block_idx, dims in block_dims.items():
                    key = (layer_idx, head_idx, block_idx)
                    selected_dims[key] = dims
    
    return selected_dims, n_layers, n_heads

def analyze_matching_rate(selected_dims, stats_json):
    """
    Compare selected dims with actual runtime argmax dims.
    Calculate matching rate.
    """
    with open(stats_json) as f:
        data = json.load(f)
    
    results = []
    
    for key, stats in data['blocks'].items():
        layer = stats['layer']
        head = stats['head']
        block = stats['block']
        
        block_key = (layer, head, block)
        
        # Get statically selected dims for this block
        selected = selected_dims.get(block_key, set())
        
        # Get runtime argmax dims (all available, sorted by frequency)
        top_argmax = stats['top_argmax_dims']  # [(dim, count), ...]
        
        if not top_argmax:
            continue
        
        # Top-1 argmax dim (most frequent)
        top1_dim = top_argmax[0][0]
        top1_count = top_argmax[0][1]
        
        # Top-K argmax dims (match PRS selection count)
        k = len(selected) if selected else 8
        topk_dims = set([d for d, c in top_argmax[:k]])
        
        # Calculate overlap
        if selected:
            overlap_dims = selected & topk_dims
            overlap_count = len(overlap_dims)
            overlap_ratio = overlap_count / k if k > 0 else 0
        else:
            overlap_dims = set()
            overlap_count = 0
            overlap_ratio = 0
        
        # Check matching
        top1_matched = top1_dim in selected if selected else False
        
        # Coverage: what % of runtime argmax occurrences are covered by selected dims
        total_tokens = stats['n_tokens']
        covered_tokens = sum(c for d, c in top_argmax if d in selected) if selected else 0
        coverage = covered_tokens / total_tokens if total_tokens > 0 else 0
        
        results.append({
            'key': key,
            'layer': layer,
            'head': head,
            'block': block,
            'selected_count': len(selected),
            'selected_dims': sorted(list(selected)) if selected else [],
            'top1_dim': top1_dim,
            'top1_count': top1_count,
            'top1_matched': top1_matched,
            'topk_dims': sorted(list(topk_dims)),
            'topk_count': k,
            'overlap_dims': sorted(list(overlap_dims)),
            'overlap_count': overlap_count,
            'overlap_ratio': overlap_ratio,
            'coverage': coverage,
            'ratio': stats['absmax/p99.9']
        })
    
    return results

def print_summary(results):
    """Print matching rate summary"""
    
    total_blocks = len(results)
    
    # Overall metrics
    blocks_with_selection = [r for r in results if r['selected_count'] > 0]
    blocks_without_selection = [r for r in results if r['selected_count'] == 0]
    
    top1_matches = sum(1 for r in blocks_with_selection if r['top1_matched'])
    avg_coverage = np.mean([r['coverage'] for r in blocks_with_selection]) if blocks_with_selection else 0
    
    print("=" * 80)
    print("PRS Matching Rate Analysis")
    print("=" * 80)
    print(f"Total blocks: {total_blocks}")
    print(f"  With PRS selection: {len(blocks_with_selection)} ({len(blocks_with_selection)/total_blocks*100:.1f}%)")
    print(f"  Without PRS selection: {len(blocks_without_selection)} ({len(blocks_without_selection)/total_blocks*100:.1f}%)")
    print()
    
    if blocks_with_selection:
        avg_overlap = np.mean([r['overlap_ratio'] for r in blocks_with_selection])
        
        print(f"[Blocks with PRS Selection: {len(blocks_with_selection)}]")
        print(f"  Top-1 argmax matched: {top1_matches}/{len(blocks_with_selection)} ({top1_matches/len(blocks_with_selection)*100:.1f}%)")
        print(f"  Average top-K overlap ratio (PRS top-8 vs runtime top-8): {avg_overlap*100:.1f}%")
        print(f"  Average coverage (selected dims cover runtime argmax): {avg_coverage*100:.1f}%")
        print()
        
        # By ratio range
        high_ratio = [r for r in blocks_with_selection if r['ratio'] > 1.3]
        if high_ratio:
            high_top1 = sum(1 for r in high_ratio if r['top1_matched'])
            high_overlap = np.mean([r['overlap_ratio'] for r in high_ratio])
            high_cov = np.mean([r['coverage'] for r in high_ratio])
            print(f"[High-Ratio Blocks (ratio>1.3): {len(high_ratio)}]")
            print(f"  Top-1 matched: {high_top1}/{len(high_ratio)} ({high_top1/len(high_ratio)*100:.1f}%)")
            print(f"  Average top-K overlap: {high_overlap*100:.1f}%")
            print(f"  Average coverage: {high_cov*100:.1f}%")
            print()
        
        # Show worst mismatches
        sorted_by_coverage = sorted(blocks_with_selection, key=lambda x: x['coverage'])
        print(f"[Worst 10 Mismatches (low coverage)]")
        print(f"{'Key':<15} {'Selected':<20} {'Top1 Dim':<10} {'Match':<8} {'Coverage':<10} {'Ratio':<8}")
        print("-" * 80)
        for r in sorted_by_coverage[:10]:
            selected_str = str(r['selected_dims'][:3]) if len(r['selected_dims']) <= 3 else str(r['selected_dims'][:2]) + "..."
            match_str = "✓" if r['top1_matched'] else "✗"
            print(f"{r['key']:<15} {selected_str:<20} {r['top1_dim']:<10} {match_str:<8} {r['coverage']*100:<9.1f}% {r['ratio']:<8.3f}")
        print()
        
        # Show best matches
        sorted_by_coverage_desc = sorted(blocks_with_selection, key=lambda x: x['coverage'], reverse=True)
        print(f"[Best 10 Matches (high coverage)]")
        print(f"{'Key':<15} {'Selected':<20} {'Top1 Dim':<10} {'Match':<8} {'Coverage':<10} {'Ratio':<8}")
        print("-" * 80)
        for r in sorted_by_coverage_desc[:10]:
            selected_str = str(r['selected_dims'][:3]) if len(r['selected_dims']) <= 3 else str(r['selected_dims'][:2]) + "..."
            match_str = "✓" if r['top1_matched'] else "✗"
            print(f"{r['key']:<15} {selected_str:<20} {r['top1_dim']:<10} {match_str:<8} {r['coverage']*100:<9.1f}% {r['ratio']:<8.3f}")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Analyze PRS matching rate')
    parser.add_argument('scales_file', help='PRS scales binary file')
    parser.add_argument('stats_json', help='Block percentile stats JSON file')
    parser.add_argument('--output', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Read PRS scales
    print("Reading PRS scales...")
    selected_dims, n_layers, n_heads = read_prs_scales(args.scales_file)
    print(f"Found {len(selected_dims)} blocks with PRS selection")
    print()
    
    # Analyze matching
    print("Analyzing matching rate...")
    results = analyze_matching_rate(selected_dims, args.stats_json)
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
