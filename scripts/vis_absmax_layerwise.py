#!/usr/bin/env python3
import struct
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

MAGIC_APOR = 0x524F5041  # "APOR"
MAGIC_ROPV = 0x524F5056  # "ROPV"

def load_absmax(file_path):
    """Load AbsMax from .bin file (APOR or ROPV format)"""
    print(f"Loading {file_path}...")
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_APOR and magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}, expected {hex(MAGIC_APOR)} or {hex(MAGIC_ROPV)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        
        print(f"  Header: layers={n_layers}, heads={n_heads}, dims={n_dims}")
        
        # Structure: [layer][head] -> {pre: array, post: array}
        data = []
        
        for layer in range(n_layers):
            layer_data = []
            for head in range(n_heads):
                # Read pre
                pre_vals = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                # Read post
                post_vals = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                
                layer_data.append({
                    'pre': pre_vals,
                    'post': post_vals
                })
            data.append(layer_data)
            
        return {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'data': data
        }

def plot_layer(layer_idx, layer_data, n_heads, n_dims, output_dir, swap_labels=False):
    """Plot 8 heads for a single layer"""
    
    # 2 rows, 4 columns for 8 heads
    n_rows = 2
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    dims = np.arange(n_dims)
    
    # Global max for y-limit consistency within layer
    max_val = 0
    for h in range(n_heads):
        if swap_labels:
            max_val = max(max_val, np.max(layer_data[h]['pre']))
        else:
            max_val = max(max_val, np.max(layer_data[h]['post']))
    
    y_limit = max_val * 1.1
    
    for h in range(n_heads):
        ax = axes[h]
        
        pre_vals = layer_data[h]['pre']
        post_vals = layer_data[h]['post']
        
        # Plot lines
        if swap_labels:
            # pre field is Post-RoPE
            ax.plot(dims, pre_vals, label='Post-RoPE', color='blue', alpha=0.6, linewidth=1)
        else:
            # post field is Post-RoPE
            ax.plot(dims, post_vals, label='Post-RoPE', color='blue', alpha=0.6, linewidth=1)
        
        # Identify outliers (simple threshold for visualization: > 50% of max)
        # Maybe highlight top values? No, keep it simple as requested.
        
        ax.set_title(f'Head {h}', fontsize=10, weight='bold')
        ax.grid(True, alpha=0.3)
        
        if h == 0:
            ax.legend(fontsize=8)
            
    # Hide empty subplots if any (shouldn't be for 8 heads)
    for i in range(n_heads, len(axes)):
        axes[i].axis('off')
        
    fig.supxlabel('Dimension', fontsize=12)
    fig.supylabel('AbsMax Value', fontsize=12)
    fig.suptitle(f'Layer {layer_idx}: Post RoPE AbsMax Distribution', fontsize=16, y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    output_path = os.path.join(output_dir, f'layer_{layer_idx:02d}.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize AbsMax distribution per layer (8 heads)')
    parser.add_argument('input_file', help='Path to rope_absmax_wikitrain.bin')
    parser.add_argument('--output-dir', '-o', required=True, help='Directory to save images')
    parser.add_argument('--layer', type=int, default=None, help='Specific layer to plot (default: all)')
    parser.add_argument('--swap-labels', action='store_true', help='Treat "pre" field as Post-RoPE (for files collected without pre-RoPE)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    data_info = load_absmax(args.input_file)
    
    n_layers = data_info['n_layers']
    n_heads = data_info['n_heads']
    n_dims = data_info['n_dims']
    all_data = data_info['data']
    
    if args.layer is not None:
        if 0 <= args.layer < n_layers:
            plot_layer(args.layer, all_data[args.layer], n_heads, n_dims, args.output_dir, swap_labels=args.swap_labels)
        else:
            print(f"Error: Layer {args.layer} out of range (0-{n_layers-1})")
    else:
        for l in range(n_layers):
            plot_layer(l, all_data[l], n_heads, n_dims, args.output_dir, swap_labels=args.swap_labels)
            
    print(f"\nDone! All images saved to {args.output_dir}")

if __name__ == '__main__':
    main()
