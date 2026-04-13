#!/usr/bin/env python3
"""
Find the pair with maximum gap between even and odd dimension values
"""

import struct
import numpy as np
import argparse

MAGIC_APOR = 0x524F5041
MAGIC_SCRS = 0x53435253

def load_absmax(file_path):
    """Load AbsMax from .bin file"""
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_APOR:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        
        pre_absmax = []
        post_absmax = []
        
        for layer in range(n_layers):
            layer_pre = []
            layer_post = []
            for head in range(n_heads):
                data_pre = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                layer_pre.append(data_pre)
                data_post = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                layer_post.append(data_post)
            
            pre_absmax.append(np.array(layer_pre))
            post_absmax.append(np.array(layer_post))
            
        return {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'pre_absmax': pre_absmax,
            'post_absmax': post_absmax
        }

def load_scales(scales_file):
    """Load CRS/PRS scales"""
    with open(scales_file, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_SCRS:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        top_k = struct.unpack('I', f.read(4))[0]

        # Detect format
        header_bytes = 4 * 6
        cur_pos = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(cur_pos, 0)

        remaining = file_size - header_bytes
        per_head_bytes_prs = ((top_k * 2) * 4) + ((top_k * 2) * 4)
        expected_prs = n_layers * n_heads * per_head_bytes_prs

        if remaining == expected_prs:
            stored_len = top_k * 2
        else:
            stored_len = top_k
        
        scales_data = []
        for layer_idx in range(n_layers):
            layer_scales = []
            for head_idx in range(n_heads):
                indices = np.frombuffer(f.read(stored_len * 4), dtype=np.int32)
                scales = np.frombuffer(f.read(stored_len * 4), dtype=np.float32)

                valid_mask = indices >= 0
                layer_scales.append({
                    'indices': indices[valid_mask],
                    'scales': scales[valid_mask]
                })
            scales_data.append(layer_scales)
        
        return scales_data

def find_max_gap_pair(absmax_data, scales_data):
    """Find pair with maximum gap between even and odd values"""
    n_layers = absmax_data['n_layers']
    n_heads = absmax_data['n_heads']
    
    max_ratio = 0
    max_info = None
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            pre_absmax = absmax_data['pre_absmax'][layer_idx][head_idx]
            outlier_indices = scales_data[layer_idx][head_idx]['indices']
            
            # Group into pairs
            pairs = {}
            for idx in outlier_indices:
                even_idx = int(idx & ~1)
                odd_idx = int(idx | 1)
                pair_key = (even_idx, odd_idx)
                
                if pair_key not in pairs:
                    even_val = pre_absmax[even_idx]
                    odd_val = pre_absmax[odd_idx]
                    pairs[pair_key] = (even_val, odd_val)
            
            # Calculate gap for each pair
            for (even_idx, odd_idx), (even_val, odd_val) in pairs.items():
                # Ratio of max to min
                max_val = max(even_val, odd_val)
                min_val = min(even_val, odd_val)
                
                if min_val > 0:
                    ratio = max_val / min_val
                    
                    if ratio > max_ratio:
                        max_ratio = ratio
                        max_info = {
                            'layer': layer_idx,
                            'head': head_idx,
                            'even_dim': even_idx,
                            'odd_dim': odd_idx,
                            'even_val': even_val,
                            'odd_val': odd_val,
                            'ratio': ratio
                        }
    
    return max_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('absmax_file', help='rope_absmax file')
    parser.add_argument('scales_file', help='scales file')
    args = parser.parse_args()
    
    print("Loading absmax data...")
    absmax_data = load_absmax(args.absmax_file)
    
    print("Loading scales data...")
    scales_data = load_scales(args.scales_file)
    
    print("\nFinding pair with maximum gap...")
    max_gap = find_max_gap_pair(absmax_data, scales_data)
    
    if max_gap:
        print("\n" + "="*70)
        print("Pair with Maximum Gap")
        print("="*70)
        print(f"Layer: {max_gap['layer']}")
        print(f"Head: {max_gap['head']}")
        print(f"Dimensions: {max_gap['even_dim']}, {max_gap['odd_dim']}")
        print(f"\nValues:")
        print(f"  dim {max_gap['even_dim']}: {max_gap['even_val']:.4f}")
        print(f"  dim {max_gap['odd_dim']}: {max_gap['odd_val']:.4f}")
        print(f"\nGap Ratio: {max_gap['ratio']:.2f}x")
        print(f"(max/min = {max(max_gap['even_val'], max_gap['odd_val']):.4f} / {min(max_gap['even_val'], max_gap['odd_val']):.4f})")
    else:
        print("No pairs found!")

if __name__ == '__main__':
    main()
