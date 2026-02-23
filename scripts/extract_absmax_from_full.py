#!/usr/bin/env python3
"""
Extract channel-wise absmax from rope_values_full.bin (variable-length format)

File format:
  Header: magic(4) version(4) layers(4) heads(4) dims(4) tokens(4)
  For each layer:
    pre_count (uint32)
    pre_values[pre_count] (float32)
    post_count (uint32)
    post_values[post_count] (float32)

Output: rope_absmax_full.bin (APOR magic)
"""

import struct
import numpy as np
import argparse
from pathlib import Path

MAGIC_ROPV = 0x524F5056  # "ROPV"
MAGIC_APOR = 0x524F5041  # "APOR"


def read_rope_values(file_path):
    """Read rope_values_full.bin with variable-length format"""
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        n_tokens = struct.unpack('I', f.read(4))[0]
        
        print(f"Header: layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")
        
        data = {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'pre_absmax': [],   # [layer][head][dim]
            'post_absmax': []   # [layer][head][dim]
        }
        
        for layer in range(n_layers):
            # Read pre-RoPE values
            pre_count = struct.unpack('I', f.read(4))[0]
            if pre_count > 0:
                pre_values = np.frombuffer(f.read(pre_count * 4), dtype=np.float32)
                n_tokens_layer = pre_count // (n_heads * n_dims)
                pre_values = pre_values.reshape(n_tokens_layer, n_heads, n_dims)
                # Channel-wise absmax per head -> Max over Tokens (axis 0)
                pre_absmax = np.abs(pre_values).max(axis=0)  # [n_heads, n_dims]
            else:
                pre_absmax = np.zeros((n_heads, n_dims), dtype=np.float32)
            
            # Read post-RoPE values
            post_count = struct.unpack('I', f.read(4))[0]
            if post_count > 0:
                post_values = np.frombuffer(f.read(post_count * 4), dtype=np.float32)
                n_tokens_layer = post_count // (n_heads * n_dims)
                post_values = post_values.reshape(n_tokens_layer, n_heads, n_dims)
                # Channel-wise absmax per head -> Max over Tokens (axis 0)
                post_absmax = np.abs(post_values).max(axis=0)  # [n_heads, n_dims]
            else:
                post_absmax = np.zeros((n_heads, n_dims), dtype=np.float32)
            
            data['pre_absmax'].append(pre_absmax)
            data['post_absmax'].append(post_absmax)
            
            if layer % 8 == 0:
                tokens_pre = pre_count // (n_heads * n_dims) if pre_count > 0 else 0
                tokens_post = post_count // (n_heads * n_dims) if post_count > 0 else 0
                print(f"  Layer {layer}: pre={tokens_pre} tokens, post={tokens_post} tokens")
        
        print(f"Done reading {n_layers} layers")
        return data


def save_absmax(data, output_path):
    """Save absmax values to binary file"""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('I', MAGIC_APOR))
        f.write(struct.pack('I', 1))  # version
        f.write(struct.pack('I', data['n_layers']))
        f.write(struct.pack('I', data['n_heads']))
        f.write(struct.pack('I', data['n_dims']))
        
        for layer in range(data['n_layers']):
            for head in range(data['n_heads']):
                f.write(data['pre_absmax'][layer][head].astype(np.float32).tobytes())
                f.write(data['post_absmax'][layer][head].astype(np.float32).tobytes())
    
    print(f"Saved to: {output_path} ({Path(output_path).stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()
    
    print(f"Reading {args.input_file}...")
    data = read_rope_values(args.input_file)
    save_absmax(data, args.output_file)
    
    # Quick analysis
    print("\n=== Pre-RoPE Absmax (Layer 8, Head 0) ===")
    pre = data['pre_absmax'][8][0]
    top_idx = np.argsort(pre)[-8:][::-1]
    print(f"Top-8 channels: {top_idx}")
    print(f"Top-8 values: {pre[top_idx]}")
    
    print("\n=== Post-RoPE Absmax (Layer 8, Head 0) ===")
    post = data['post_absmax'][8][0]
    top_idx = np.argsort(post)[-8:][::-1]
    print(f"Top-8 channels: {top_idx}")
    print(f"Top-8 values: {post[top_idx]}")


if __name__ == '__main__':
    main()
