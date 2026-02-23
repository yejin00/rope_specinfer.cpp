#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import struct

MAGIC_ROPV = 0x524F5056  # "ROPV"

def read_activation_dump(file_path, target_layer, n_head_kv=8, head_dim=128):
    """
    Reads variable-length ROPV format activation dump.
    Returns data for specified layer: [n_tokens, n_heads, n_dims]
    """
    with open(file_path, 'rb') as f:
        # Read header
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: 0x{magic:08X}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        n_tokens = struct.unpack('I', f.read(4))[0]
        
        print(f"Loading {file_path}...")
        print(f"  ROPV format: layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")
        
        # Read through layers
        for layer_idx in range(n_layers):
            # Read pre-RoPE count and data
            pre_count = struct.unpack('I', f.read(4))[0]
            
            if layer_idx == target_layer:
                if pre_count == 0:
                    print(f"  Layer {layer_idx}: No data")
                    return None
                
                # Read pre-RoPE values (float32)
                pre_values = np.frombuffer(f.read(pre_count * 4), dtype=np.float32)
                n_tokens_layer = pre_count // (n_heads * n_dims)
                data = pre_values.reshape(n_tokens_layer, n_heads, n_dims)
                
                # Skip post-RoPE
                post_count = struct.unpack('I', f.read(4))[0]
                if post_count > 0:
                    f.seek(post_count * 4, 1)
                
                print(f"  Layer {layer_idx}: {n_tokens_layer} tokens")
                return data
            else:
                # Skip this layer
                if pre_count > 0:
                    f.seek(pre_count * 4, 1)
                post_count = struct.unpack('I', f.read(4))[0]
                if post_count > 0:
                    f.seek(post_count * 4, 1)
        
        return None

def plot_histogram(data, layer_idx, head_idx, dim_idx, output_path):
    # Extract specific channel: [n_tokens]
    channel_data = data[:, head_idx, dim_idx]
    
    # Statistics
    min_val = np.min(channel_data)
    max_val = np.max(channel_data)
    mean_val = np.mean(channel_data)
    std_val = np.std(channel_data)
    absmax = np.max(np.abs(channel_data))
    
    print(f"\n[Statistics for L{layer_idx} H{head_idx} D{dim_idx}]")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")
    print(f"  Mean: {mean_val:.4f}")
    print(f"  Std: {std_val:.4f}")
    print(f"  AbsMax: {absmax:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(channel_data, bins=100, color='skyblue', edgecolor='black', alpha=0.7, log=True)
    
    plt.title(f'Activation Histogram: Layer {layer_idx}, Head {head_idx}, Dim {dim_idx}')
    plt.xlabel('Activation Value')
    plt.ylabel('Count (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # Add text box with stats
    stats_text = (f'Min: {min_val:.2f}\n'
                  f'Max: {max_val:.2f}\n'
                  f'Mean: {mean_val:.2f}\n'
                  f'AbsMax: {absmax:.2f}')
    
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot activation histogram from dump file')
    parser.add_argument('file_path', help='Path to ROPV file')
    parser.add_argument('--layer', type=int, required=True, help='Layer index')
    parser.add_argument('--head', type=int, required=True, help='Head index (0-7)')
    parser.add_argument('--dim', type=int, required=True, help='Dimension index (0-127)')
    parser.add_argument('--output', '-o', required=True, help='Output image path')
    parser.add_argument('--n-head-kv', type=int, default=8, help='Number of KV heads')
    parser.add_argument('--head-dim', type=int, default=128, help='Head dimension')
    
    args = parser.parse_args()
    
    data = read_activation_dump(args.file_path, args.layer, args.n_head_kv, args.head_dim)
    if data is not None:
        plot_histogram(data, args.layer, args.head, args.dim, args.output)

if __name__ == '__main__':
    main()
