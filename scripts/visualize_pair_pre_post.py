#!/usr/bin/env python3
"""
Visualize token-by-token values for a specific pair of channels from rope_values_full.bin.
Generates separate graphs for Pre-RoPE and Post-RoPE values.

Usage:
    python scripts/visualize_pair_pre_post.py <rope_values_full.bin> <layer> <head> <dim1> <dim2> [start_token] [end_token] [output_dir]
"""

import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_rope_values_full_header(filepath):
    """Read header from rope_values_full.bin file"""
    with open(filepath, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        layers = struct.unpack('I', f.read(4))[0]
        heads = struct.unpack('I', f.read(4))[0]
        dims = struct.unpack('I', f.read(4))[0]
        tokens = struct.unpack('I', f.read(4))[0]
        
        print(f"Magic: {hex(magic)}")
        print(f"Version: {version}")
        print(f"Layers: {layers}")
        print(f"Heads: {heads}")
        print(f"Dims (head_dim): {dims}")
        print(f"Tokens: {tokens}")
        
        if magic != 0x524F5056:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected 0x524F5056")
        
        header_size = 24  # 6 * 4 bytes
        
        return {
            'magic': magic,
            'version': version,
            'layers': layers,
            'heads': heads,
            'dims': dims,
            'tokens': tokens,
            'header_size': header_size
        }

def read_channel_values_full(filepath, header, layer, head, dim, start_token=0, end_token=None):
    """
    Read values for a specific channel from rope_values_full.bin.
    
    File layout:
    - Header (24 bytes)
    - For each layer:
        - pre_count (uint32)
        - pre_values[pre_count] (float array: tokens * heads * dims)
        - post_count (uint32)
        - post_values[post_count] (float array: tokens * heads * dims)
    
    Data within pre/post_values is token-major:
        For each token:
            For each head:
                For each dim:
                    float value
    
    Returns: (pre_values, post_values)
    """
    layers = header['layers']
    heads = header['heads']
    dims = header['dims']
    total_tokens = header['tokens']
    
    if end_token is None or end_token > total_tokens:
        end_token = total_tokens
    
    start_token = max(0, start_token)
    end_token = min(end_token, total_tokens)
    num_tokens = end_token - start_token
    
    pre_values = []
    post_values = []
    
    with open(filepath, 'rb') as f:
        f.seek(header['header_size'])
        
        # Skip to target layer
        for l in range(layer):
            # Read and skip pre_values
            pre_count = struct.unpack('I', f.read(4))[0]
            f.seek(pre_count * 4, 1)  # skip pre_values
            
            # Read and skip post_values
            post_count = struct.unpack('I', f.read(4))[0]
            f.seek(post_count * 4, 1)  # skip post_values
        
        # Read target layer's pre_values
        pre_count = struct.unpack('I', f.read(4))[0]
        expected_count = total_tokens * heads * dims
        
        if pre_count != expected_count:
            print(f"Warning: pre_count={pre_count}, expected={expected_count}")
        
        # Read all pre_values for this layer
        all_pre = struct.unpack(f'{pre_count}f', f.read(pre_count * 4))
        
        # Extract specific channel for token range
        for token_idx in range(start_token, end_token):
            # Index in flattened array: token * (heads * dims) + head * dims + dim
            idx = token_idx * (heads * dims) + head * dims + dim
            if idx < len(all_pre):
                pre_values.append(all_pre[idx])
            else:
                pre_values.append(0.0)
        
        # Read target layer's post_values
        post_count = struct.unpack('I', f.read(4))[0]
        
        if post_count != expected_count:
            print(f"Warning: post_count={post_count}, expected={expected_count}")
        
        # Read all post_values for this layer
        all_post = struct.unpack(f'{post_count}f', f.read(post_count * 4))
        
        # Extract specific channel for token range
        for token_idx in range(start_token, end_token):
            idx = token_idx * (heads * dims) + head * dims + dim
            if idx < len(all_post):
                post_values.append(all_post[idx])
            else:
                post_values.append(0.0)
    
    return pre_values, post_values

def plot_pair_values_separate(pre_values1, pre_values2, post_values1, post_values2, 
                               layer, head, dim1, dim2, output_dir, start_token=0):
    """Plot two separate graphs: one for pre-RoPE, one for post-RoPE"""
    num_tokens = len(pre_values1)
    tokens = np.arange(start_token, start_token + num_tokens)
    
    # Convert to absolute values
    abs_pre1 = np.abs(pre_values1)
    abs_pre2 = np.abs(pre_values2)
    abs_post1 = np.abs(post_values1)
    abs_post2 = np.abs(post_values2)
    
    # Pre-RoPE graph
    plt.figure(figsize=(12, 6))
    plt.plot(tokens, abs_pre1, 'o-', label=f'Dim {dim1}', alpha=0.7, markersize=6, color='blue')
    plt.plot(tokens, abs_pre2, 's-', label=f'Dim {dim2}', alpha=0.7, markersize=6, color='orange')
    plt.xlabel('Token Index', fontsize=12)
    plt.ylabel('Absolute Value', fontsize=12)
    plt.title(f'Pre-RoPE: Layer {layer}, Head {head}, Dim {dim1} vs Dim {dim2}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    pre_output = f"{output_dir}/pre_L{layer}_H{head}_D{dim1}_D{dim2}_T{start_token}-{start_token+num_tokens}.png"
    plt.savefig(pre_output, dpi=150, bbox_inches='tight')
    print(f"Saved Pre-RoPE: {pre_output}")
    plt.close()
    
    # Post-RoPE graph
    plt.figure(figsize=(12, 6))
    plt.plot(tokens, abs_post1, 'o-', label=f'Dim {dim1}', alpha=0.7, markersize=6, color='blue')
    plt.plot(tokens, abs_post2, 's-', label=f'Dim {dim2}', alpha=0.7, markersize=6, color='orange')
    plt.xlabel('Token Index', fontsize=12)
    plt.ylabel('Absolute Value', fontsize=12)
    plt.title(f'Post-RoPE: Layer {layer}, Head {head}, Dim {dim1} vs Dim {dim2}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    post_output = f"{output_dir}/post_L{layer}_H{head}_D{dim1}_D{dim2}_T{start_token}-{start_token+num_tokens}.png"
    plt.savefig(post_output, dpi=150, bbox_inches='tight')
    print(f"Saved Post-RoPE: {post_output}")
    plt.close()

def main():
    if len(sys.argv) < 6:
        print("Usage: python visualize_pair_pre_post.py <rope_values_full.bin> <layer> <head> <dim1> <dim2> [start_token] [end_token] [output_dir]")
        print("\nExample:")
        print("  python scripts/visualize_pair_pre_post.py rope_values_full.bin 0 0 0 1 0 100")
        print("  python scripts/visualize_pair_pre_post.py rope_values_full.bin 5 3 10 11 100 200 output/")
        sys.exit(1)
    
    filepath = sys.argv[1]
    layer = int(sys.argv[2])
    head = int(sys.argv[3])
    dim1 = int(sys.argv[4])
    dim2 = int(sys.argv[5])
    start_token = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    end_token = int(sys.argv[7]) if len(sys.argv) > 7 else start_token + 100
    output_dir = sys.argv[8] if len(sys.argv) > 8 else str(Path(filepath).parent)
    
    print(f"Reading: {filepath}")
    print(f"Target: Layer {layer}, Head {head}, Dims [{dim1}, {dim2}]")
    print(f"Extracting tokens {start_token} to {end_token}")
    
    try:
        header = read_rope_values_full_header(filepath)
    except Exception as e:
        print(f"Error reading header: {e}")
        sys.exit(1)
    
    # Validate inputs
    if layer >= header['layers']:
        print(f"Error: layer {layer} >= total layers {header['layers']}")
        sys.exit(1)
    if head >= header['heads']:
        print(f"Error: head {head} >= total heads {header['heads']}")
        sys.exit(1)
    if dim1 >= header['dims'] or dim2 >= header['dims']:
        print(f"Error: dim out of range (max: {header['dims']-1})")
        sys.exit(1)
    
    print("\nExtracting channel values...")
    
    # Read dim1
    pre_values1, post_values1 = read_channel_values_full(filepath, header, layer, head, dim1, start_token, end_token)
    
    # Read dim2
    pre_values2, post_values2 = read_channel_values_full(filepath, header, layer, head, dim2, start_token, end_token)
    
    print(f"Pre-RoPE Dim {dim1} values: {pre_values1[:5]}...")
    print(f"Pre-RoPE Dim {dim2} values: {pre_values2[:5]}...")
    print(f"Post-RoPE Dim {dim1} values: {post_values1[:5]}...")
    print(f"Post-RoPE Dim {dim2} values: {post_values2[:5]}...")
    
    # Plot both graphs
    plot_pair_values_separate(pre_values1, pre_values2, post_values1, post_values2,
                              layer, head, dim1, dim2, output_dir, start_token)
    
    print("\nDone! Generated 2 graphs (pre and post RoPE).")

if __name__ == '__main__':
    main()
