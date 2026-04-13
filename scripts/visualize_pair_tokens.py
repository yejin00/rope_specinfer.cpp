#!/usr/bin/env python3
"""
Visualize token-by-token values for a specific pair of channels.
Shows how two channels' values change across tokens.

Usage:
    python scripts/visualize_pair_tokens.py <rope_values.bin> <layer> <head> <dim1> <dim2> [start_token] [end_token] [output_dir]
"""

import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_rope_values_header(filepath):
    """Read header from rope_values.bin file"""
    with open(filepath, 'rb') as f:
        # Try to infer format - typically: magic, version, layers, heads, dims, tokens
        magic = struct.unpack('I', f.read(4))[0]
        
        # Check if it's a rope_values file (might have different magic)
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

def read_channel_values(filepath, header, layer, head, dim, start_token=0, end_token=None):
    """
    Read values for a specific channel (layer, head, dim) across tokens.
    
    Data layout (assuming token-major):
    For each token:
        For each layer:
            For each head:
                For each dim:
                    float value
    
    Or could be layer-major, head-major, etc. We'll try token-major first.
    """
    layers = header['layers']
    heads = header['heads']
    dims = header['dims']
    total_tokens = header['tokens']
    
    if end_token is None or end_token > total_tokens:
        end_token = total_tokens
    
    start_token = max(0, start_token)
    end_token = min(end_token, total_tokens)
    
    values = []
    
    with open(filepath, 'rb') as f:
        f.seek(header['header_size'])
        
        # Try token-major layout
        for token_idx in range(start_token, end_token):
            # Calculate offset for this token
            token_offset = token_idx * layers * heads * dims * 4  # 4 bytes per float
            
            # Offset within token for specific (layer, head, dim)
            channel_offset = (layer * heads * dims + head * dims + dim) * 4
            
            f.seek(header['header_size'] + token_offset + channel_offset)
            value = struct.unpack('f', f.read(4))[0]
            values.append(value)
    
    return values

def plot_pair_values(values1, values2, layer, head, dim1, dim2, output_path, start_token=0):
    """Plot values for two channels across tokens"""
    num_tokens = len(values1)
    tokens = np.arange(start_token, start_token + num_tokens)
    
    abs_values1 = np.abs(values1)
    abs_values2 = np.abs(values2)
    
    plt.figure(figsize=(12, 6))
    plt.plot(tokens, abs_values1, 'o-', label=f'Dim {dim1}', alpha=0.7, markersize=6, color='blue')
    plt.plot(tokens, abs_values2, 's-', label=f'Dim {dim2}', alpha=0.7, markersize=6, color='orange')
    
    plt.xlabel('Token Index', fontsize=12)
    plt.ylabel('Absolute Value', fontsize=12)
    plt.title(f'Layer {layer}, Head {head}: Dim {dim1} vs Dim {dim2} across Tokens', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    if len(sys.argv) < 6:
        print("Usage: python visualize_pair_tokens.py <rope_values.bin> <layer> <head> <dim1> <dim2> [start_token] [end_token] [output_dir]")
        print("\nExample:")
        print("  python scripts/visualize_pair_tokens.py rope_values.bin 0 0 0 1 0 10")
        print("  python scripts/visualize_pair_tokens.py rope_values.bin 0 0 0 1 100 200  # tokens 100-200")
        sys.exit(1)
    
    filepath = sys.argv[1]
    layer = int(sys.argv[2])
    head = int(sys.argv[3])
    dim1 = int(sys.argv[4])
    dim2 = int(sys.argv[5])
    start_token = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    end_token = int(sys.argv[7]) if len(sys.argv) > 7 else start_token + 10
    output_dir = sys.argv[8] if len(sys.argv) > 8 else str(Path(filepath).parent)
    
    print(f"Reading: {filepath}")
    print(f"Target: Layer {layer}, Head {head}, Dims [{dim1}, {dim2}]")
    print(f"Extracting tokens {start_token} to {end_token}")
    
    try:
        header = read_rope_values_header(filepath)
    except Exception as e:
        print(f"Error reading header: {e}")
        print("Trying alternative format...")
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
    values1 = read_channel_values(filepath, header, layer, head, dim1, start_token, end_token)
    values2 = read_channel_values(filepath, header, layer, head, dim2, start_token, end_token)
    
    print(f"Dim {dim1} values: {values1[:5]}...")
    print(f"Dim {dim2} values: {values2[:5]}...")
    
    output_path = f"{output_dir}/pair_L{layer}_H{head}_D{dim1}_D{dim2}_T{start_token}-{end_token}.png"
    plot_pair_values(values1, values2, layer, head, dim1, dim2, output_path, start_token)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
