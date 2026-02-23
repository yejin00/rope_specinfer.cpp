#!/usr/bin/env python3
"""
Binary GGUF Weight Fusion - Preserves ALL metadata
Copies original GGUF, then modifies tensor data in-place
"""

import struct
import numpy as np
import argparse
import shutil
import sys
from pathlib import Path

try:
    from gguf import GGUFReader
except ImportError:
    print("Error: gguf library not found")
    sys.exit(1)


MAGIC_SCRS = 0x53435253


def load_crs_scales(scales_path):
    with open(scales_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        top_k = struct.unpack('I', f.read(4))[0]

        # PRS format: header top_k denotes number of pairs, but data stores 2*top_k dims.
        # Legacy CRS format: header top_k equals number of dims stored.
        # We infer which one by checking remaining file size.
        header_bytes = 4 * 6
        cur_pos = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(cur_pos, 0)

        remaining = file_size - header_bytes
        per_head_bytes_crs = (top_k * 4) + (top_k * 4)
        per_head_bytes_prs = ((top_k * 2) * 4) + ((top_k * 2) * 4)
        expected_crs = n_layers * n_heads * per_head_bytes_crs
        expected_prs = n_layers * n_heads * per_head_bytes_prs

        if remaining == expected_prs:
            stored_len = top_k * 2
            fmt = 'PRS'
        elif remaining == expected_crs:
            stored_len = top_k
            fmt = 'CRS'
        else:
            # Default to PRS (newer) to avoid silently reading half the data.
            stored_len = top_k * 2
            fmt = 'PRS?'

        print(
            f"CRS Scales ({fmt}): {n_layers} layers, {n_heads} heads, {n_dims} dims, "
            f"k={top_k} ({'pairs' if stored_len == top_k * 2 else 'dims'}), stored_len={stored_len}"
        )
        
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
        
        return {'n_layers': n_layers, 'n_heads': n_heads, 'n_dims': n_dims, 'scales': scales_data}


def apply_crs(weight, layer_idx, scales_data, wtype):
    """Apply CRS to weight matrix"""
    n_heads = scales_data['n_heads']
    n_dims = scales_data['n_dims']
    layer_scales = scales_data['scales'][layer_idx]
    
    for head_idx, head_data in enumerate(layer_scales):
        indices = head_data['indices']
        scales = head_data['scales']
        
        if len(indices) == 0:
            continue
        
        base = head_idx * n_dims
        
        if wtype == 'k':
            # WK: divide by scale
            # Weight shape is [out_features, in_features]
            # We want to scale output features (rows) corresponding to head dims
            for idx, scale in zip(indices, scales):
                row = base + idx
                if row < weight.shape[0]:
                    weight[row, :] /= scale
        
        elif wtype == 'q':
            # WQ: multiply by scale (handle GQA)
            # WQ typically has more heads than WK in GQA
            n_q_heads = weight.shape[0] // n_dims  # Use shape[0] for out_features
            n_kv_heads = n_heads
            
            if n_kv_heads == 0: continue
            
            groups = n_q_heads // n_kv_heads
            
            for g in range(groups):
                q_head = head_idx * groups + g
                q_base = q_head * n_dims
                
                for idx, scale in zip(indices, scales):
                    row = q_base + idx
                    if row < weight.shape[0]:
                        weight[row, :] *= scale
    
    return weight


def main():
    parser = argparse.ArgumentParser(description='Binary GGUF CRS Fusion (preserves metadata)')
    parser.add_argument('input_gguf', help='Input GGUF file')
    parser.add_argument('scales_file', help='CRS scales file')
    parser.add_argument('-o', '--output', required=True, help='Output GGUF file')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Binary GGUF CRS Weight Fusion")
    print("="*60)
    
    # Load scales
    print("\n[1/4] Loading CRS scales...")
    scales = load_crs_scales(args.scales_file)
    
    # Copy original file
    print(f"\n[2/4] Copying {args.input_gguf} → {args.output}...")
    shutil.copy2(args.input_gguf, args.output)
    print(f"  Copied {Path(args.input_gguf).stat().st_size / (1024**3):.2f} GB")
    
    # Read tensor info from original
    print("\n[3/4] Reading tensor information...")
    reader = GGUFReader(args.input_gguf)
    
    # Find Q/K tensors and their file offsets
    tensors_to_modify = []
    
    for tensor in reader.tensors:
        is_q = '.attn_q.weight' in tensor.name
        is_k = '.attn_k.weight' in tensor.name
        
        if is_q or is_k:
            parts = tensor.name.split('.')
            if len(parts) > 1 and parts[0] == 'blk':
                layer_idx = int(parts[1])
                wtype = 'q' if is_q else 'k'
                
                tensors_to_modify.append({
                    'name': tensor.name,
                    'layer': layer_idx,
                    'type': wtype,
                    'offset': tensor.data_offset,
                    'shape': tensor.shape,
                    'dtype': tensor.tensor_type,
                    'original_data': tensor.data
                })
    
    print(f"  Found {len(tensors_to_modify)} tensors to modify")
    
    # Modify tensors in copied file
    print("\n[4/4] Modifying tensors in-place...")
    
    with open(args.output, 'r+b') as f:
        for i, info in enumerate(tensors_to_modify):
            # Load weight
            weight = info['original_data'].astype(np.float32)
            
            # Apply CRS
            weight = apply_crs(weight, info['layer'], scales, info['type'])
            
            # Convert back to original dtype
            weight = weight.astype(info['original_data'].dtype)
            
            # Write to file at correct offset
            f.seek(info['offset'])
            f.write(weight.tobytes())
            
            if (i + 1) % 10 == 0:
                print(f"  Modified {i+1}/{len(tensors_to_modify)} tensors...")
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully modified {len(tensors_to_modify)} weight tensors")
    print(f"✓ Output: {args.output}")
    print(f"✓ All metadata preserved from original file")
    print(f"{'='*60}")
    
    print(f"\nTest with:")
    print(f"  ./build/bin/llama-cli -m {args.output} -p 'Hello' -n 20")


if __name__ == '__main__':
    main()
