#!/usr/bin/env python3
"""
Static CRS Scale Extraction Script

This script analyzes Pre-RoPE key activation values to identify static outlier channels
and extract per-channel scales for offline CRS.

Stage 1: Static Outlier (Offline CRS)
- Identifies channels with persistently large values BEFORE RoPE application
- Typically found in later dimensions (80-128) - low frequency region
- These outliers are consistent across tokens and can be pre-computed

Usage:
    python extract_static_crs_scales.py <rope_values.bin> <output_scales.bin> [--top-k 8]
    
Output format (scales_static_k.bin):
    - Magic: 0x53435253 ("SCRS" - Static CRS)
    - Version: uint32
    - n_layers: uint32
    - n_heads: uint32 (head_kv)
    - n_dims: uint32 (head_dim)
    - For each layer:
        - For each head:
            - n_outlier_channels: uint32
            - outlier_indices: int32[n_outlier_channels]
            - scales: float32[n_outlier_channels]
"""

import struct
import numpy as np
import argparse
import sys
from pathlib import Path

MAGIC_ROPE = 0x524F5056  # "ROPV"
MAGIC_SCRS = 0x53435253  # "SCRS" - Static CRS
VERSION = 1


def read_rope_values(file_path):
    """Read rope_values.bin file containing pre/post RoPE values"""
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_ROPE:
            raise ValueError(f"Invalid magic number: {hex(magic)}, expected {hex(MAGIC_ROPE)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        n_tokens = struct.unpack('I', f.read(4))[0]
        
        print(f"RoPE values file:")
        print(f"  Version: {version}")
        print(f"  Layers: {n_layers}, Heads: {n_heads}, Dims: {n_dims}, Tokens: {n_tokens}")
        
        layer_data = []
        for layer in range(n_layers):
            pre_count = struct.unpack('I', f.read(4))[0]
            post_count = struct.unpack('I', f.read(4))[0]
            
            if pre_count > 0:
                pre_rope = np.frombuffer(f.read(pre_count * 4), dtype=np.float32)
                pre_rope = pre_rope.reshape(n_tokens, n_heads, n_dims)
            else:
                pre_rope = None
                
            if post_count > 0:
                post_rope = np.frombuffer(f.read(post_count * 4), dtype=np.float32)
                post_rope = post_rope.reshape(n_tokens, n_heads, n_dims)
            else:
                post_rope = None
                
            layer_data.append({
                'pre': pre_rope,
                'post': post_rope
            })
        
        return {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'n_tokens': n_tokens,
            'layers': layer_data
        }


def extract_static_outliers(data, top_k=8, verbose=True):
    """
    Extract static outlier channels from Pre-RoPE values.
    
    Static outliers are channels that consistently have large absolute values
    across all tokens, BEFORE RoPE is applied.
    
    For each layer and head:
    1. Compute channel-wise absmax across all tokens
    2. Select top-k channels with largest absmax
    3. Compute scale as absmax for those channels
    
    Returns:
        Dict with outlier info per layer/head
    """
    n_layers = data['n_layers']
    n_heads = data['n_heads']
    n_dims = data['n_dims']
    
    results = {
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_dims': n_dims,
        'top_k': top_k,
        'outliers': []  # [layer][head] = {'indices': [...], 'scales': [...]}
    }
    
    print(f"\nExtracting static outliers (top-{top_k} per head)...")
    print(f"Focus on later dimensions (high index = low frequency = stable under RoPE)")
    
    for layer_idx in range(n_layers):
        layer_outliers = []
        pre_rope = data['layers'][layer_idx]['pre']
        
        if pre_rope is None:
            print(f"  Layer {layer_idx}: No pre-RoPE data")
            layer_outliers = [{'indices': np.array([]), 'scales': np.array([])} for _ in range(n_heads)]
            results['outliers'].append(layer_outliers)
            continue
        
        for head_idx in range(n_heads):
            # Get pre-RoPE values for this head: [n_tokens, n_dims]
            head_data = pre_rope[:, head_idx, :]
            
            # Channel-wise absmax across all tokens
            channel_absmax = np.max(np.abs(head_data), axis=0)  # [n_dims]
            
            # Find top-k channels by absmax
            top_indices = np.argsort(channel_absmax)[-top_k:][::-1]  # Descending order
            top_scales = channel_absmax[top_indices]
            
            layer_outliers.append({
                'indices': top_indices.astype(np.int32),
                'scales': top_scales.astype(np.float32)
            })
            
            if verbose and layer_idx % 8 == 0 and head_idx == 0:
                print(f"  Layer {layer_idx}, Head {head_idx}:")
                print(f"    Top-{top_k} outlier channels: {top_indices}")
                print(f"    Scales (absmax): {top_scales}")
                # Check if outliers are in later dimensions (80-128)
                late_dim_count = np.sum(top_indices >= 80)
                print(f"    Channels in late dims (>=80): {late_dim_count}/{top_k}")
        
        results['outliers'].append(layer_outliers)
    
    return results


def analyze_outlier_distribution(results):
    """Analyze the distribution of static outliers across dimensions"""
    print("\n" + "="*60)
    print("Static Outlier Distribution Analysis")
    print("="*60)
    
    all_indices = []
    for layer_outliers in results['outliers']:
        for head_outliers in layer_outliers:
            all_indices.extend(head_outliers['indices'].tolist())
    
    all_indices = np.array(all_indices)
    n_dims = results['n_dims']
    
    # Divide into regions
    early = np.sum((all_indices >= 0) & (all_indices < 32))
    mid_early = np.sum((all_indices >= 32) & (all_indices < 64))
    mid_late = np.sum((all_indices >= 64) & (all_indices < 96))
    late = np.sum((all_indices >= 96) & (all_indices < n_dims))
    
    total = len(all_indices)
    print(f"\nChannel distribution (total {total} outlier selections):")
    print(f"  Early dims   [0-31]:   {early:4d} ({100*early/total:.1f}%) - High frequency (RoPE sensitive)")
    print(f"  Mid-early   [32-63]:   {mid_early:4d} ({100*mid_early/total:.1f}%)")
    print(f"  Mid-late    [64-95]:   {mid_late:4d} ({100*mid_late/total:.1f}%)")
    print(f"  Late dims  [96-127]:   {late:4d} ({100*late/total:.1f}%) - Low frequency (RoPE stable)")
    
    # Most common channels
    unique, counts = np.unique(all_indices, return_counts=True)
    top_channels = sorted(zip(unique, counts), key=lambda x: -x[1])[:10]
    print(f"\nMost frequently selected outlier channels:")
    for ch, cnt in top_channels:
        print(f"  Channel {ch:3d}: {cnt:4d} times")


def save_static_scales(results, output_path):
    """
    Save static CRS scales to binary file.
    
    Format:
        Magic: uint32 (0x53435253)
        Version: uint32
        n_layers: uint32
        n_heads: uint32
        n_dims: uint32
        top_k: uint32
        
        For each layer:
            For each head:
                indices: int32[top_k]
                scales: float32[top_k]
    """
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('I', MAGIC_SCRS))
        f.write(struct.pack('I', VERSION))
        f.write(struct.pack('I', results['n_layers']))
        f.write(struct.pack('I', results['n_heads']))
        f.write(struct.pack('I', results['n_dims']))
        f.write(struct.pack('I', results['top_k']))
        
        # Per-layer, per-head data
        for layer_idx, layer_outliers in enumerate(results['outliers']):
            for head_idx, head_outliers in enumerate(layer_outliers):
                indices = head_outliers['indices']
                scales = head_outliers['scales']
                
                # Pad to top_k if needed
                top_k = results['top_k']
                if len(indices) < top_k:
                    indices = np.pad(indices, (0, top_k - len(indices)), constant_values=-1)
                    scales = np.pad(scales, (0, top_k - len(scales)), constant_values=1.0)
                
                f.write(indices.astype(np.int32).tobytes())
                f.write(scales.astype(np.float32).tobytes())
    
    print(f"\nSaved static CRS scales to: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size} bytes")


def main():
    parser = argparse.ArgumentParser(description='Extract static CRS scales from Pre-RoPE values')
    parser.add_argument('rope_file', help='Path to rope_values.bin')
    parser.add_argument('output_file', help='Output path for static scales (scales_static_k.bin)')
    parser.add_argument('--top-k', type=int, default=8, help='Number of top outlier channels per head (default: 8)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed info')
    
    args = parser.parse_args()
    
    # Read rope values
    print(f"Reading {args.rope_file}...")
    data = read_rope_values(args.rope_file)
    
    # Extract static outliers
    results = extract_static_outliers(data, top_k=args.top_k, verbose=args.verbose)
    
    # Analyze distribution
    analyze_outlier_distribution(results)
    
    # Save scales
    save_static_scales(results, args.output_file)
    
    print("\nDone! Use this file with --cache-type-k-static-scales option")


if __name__ == '__main__':
    main()
