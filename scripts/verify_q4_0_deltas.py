#!/usr/bin/env python3
"""
Verify Q4_0 block deltas to confirm weight fusion improved quantization precision.
"""

import struct
import numpy as np
import argparse
import sys
from pathlib import Path

try:
    import gguf
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)


QK4_0 = 32


def load_q4_0_deltas(gguf_path, tensor_name):
    """Load Q4_0 block deltas for a specific tensor."""
    reader = gguf.GGUFReader(gguf_path)
    
    for tensor in reader.tensors:
        if tensor.name == tensor_name:
            if tensor.tensor_type != gguf.GGMLQuantizationType.Q4_0:
                raise ValueError(f"Tensor {tensor_name} is not Q4_0 type")
            
            # Read actual tensor data from file
            n_elements = np.prod(tensor.shape)
            n_blocks = (n_elements + QK4_0 - 1) // QK4_0
            block_size = 2 + QK4_0 // 2  # 2 bytes delta + 16 bytes qs
            
            # Open file and seek to tensor data offset
            with open(gguf_path, 'rb') as f:
                f.seek(tensor.data_offset)
                
                deltas = []
                for _ in range(n_blocks):
                    delta_bytes = f.read(2)
                    if len(delta_bytes) < 2:
                        break
                    delta = np.frombuffer(delta_bytes, dtype=np.float16)[0]
                    deltas.append(float(delta))
                    
                    # Skip quantized values
                    f.read(QK4_0 // 2)
            
            return np.array(deltas), tensor.shape
    
    raise ValueError(f"Tensor {tensor_name} not found")


def load_crs_scales(scales_path):
    """Load CRS scales."""
    MAGIC_SCRS = 0x53435253
    
    with open(scales_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_SCRS:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        k = struct.unpack('I', f.read(4))[0]

        # PRS format: header k denotes number of pairs, but each head stores 2*k indices and 2*k scales.
        # Legacy CRS format: header k equals number of dims stored.
        header_bytes = 4 * 6
        cur_pos = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(cur_pos, 0)

        remaining = file_size - header_bytes
        per_head_bytes_crs = (k * 4) + (k * 4)
        per_head_bytes_prs = ((k * 2) * 4) + ((k * 2) * 4)
        expected_crs = n_layers * n_heads * per_head_bytes_crs
        expected_prs = n_layers * n_heads * per_head_bytes_prs

        if remaining == expected_prs:
            stored_len = k * 2
        elif remaining == expected_crs:
            stored_len = k
        else:
            # Default to PRS to avoid silently reading half the data
            stored_len = k * 2
        
        scales = []
        for _ in range(n_layers):
            layer_scales = []
            for _ in range(n_heads):
                indices = np.frombuffer(f.read(stored_len * 4), dtype=np.int32)
                scale_values = np.frombuffer(f.read(stored_len * 4), dtype=np.float32)

                valid_mask = indices >= 0
                indices = indices[valid_mask]
                scale_values = scale_values[valid_mask]
                layer_scales.append({
                    'indices': indices,
                    'scales': scale_values
                })
            scales.append(layer_scales)
    
    return {
        'n_layers': n_layers,
        'n_heads': n_heads,
        'n_dims': n_dims,
        'k': k,
        'scales': scales
    }


def analyze_deltas(orig_deltas, fused_deltas, shape, layer_idx, scales_data):
    """Analyze Q4_0 deltas to verify fusion improved precision."""
    
    # Shape: [n_rows, n_cols] where n_cols = n_heads * n_dims
    n_rows, n_cols = shape
    n_heads = scales_data['n_heads']
    n_dims = scales_data['n_dims']
    
    print(f"\n{'='*70}")
    print(f"Q4_0 Delta Analysis - Layer {layer_idx}")
    print(f"{'='*70}\n")
    
    print(f"Tensor shape: {shape}")
    print(f"Total blocks: {len(orig_deltas)}")
    blocks_per_row = int((n_cols + QK4_0 - 1) // QK4_0)
    print(f"Blocks per row: {blocks_per_row}")
    print()
    
    # Analyze outlier dimensions
    layer_scales = scales_data['scales'][layer_idx]
    
    results = []
    
    for head_idx in range(n_heads):
        head_data = layer_scales[head_idx]
        indices = head_data['indices']
        scale_values = head_data['scales']
        
        for idx, scale in zip(indices, scale_values):
            if idx < 0:  # Invalid index
                continue
            
            # Column index for this dimension
            col_idx = int(head_idx * n_dims + idx)

            # Q4_0 is stored row-major: for each row, blocks cover columns in chunks of 32.
            # For a given column, we need the delta of the block that contains that column, for every row.
            block_in_row = int(col_idx // QK4_0)
            if block_in_row >= blocks_per_row:
                continue

            block_indices = (np.arange(n_rows, dtype=np.int64) * blocks_per_row) + block_in_row
            if block_indices[-1] >= len(orig_deltas):
                continue

            orig_col_deltas = orig_deltas[block_indices]
            fused_col_deltas = fused_deltas[block_indices]
            
            # Statistics
            orig_mean = float(np.mean(orig_col_deltas))
            fused_mean = float(np.mean(fused_col_deltas))
            orig_max = np.max(orig_col_deltas)
            fused_max = np.max(fused_col_deltas)
            
            # Use absolute mean because deltas can be negative depending on block scale sign
            orig_mean_abs = abs(orig_mean)
            fused_mean_abs = abs(fused_mean)
            reduction_mean = (orig_mean_abs / fused_mean_abs) if fused_mean_abs > 0 else 0.0
            reduction_max = (orig_max / fused_max) if fused_max > 0 else 0.0
            
            results.append({
                'head': head_idx,
                'dim': idx,
                'scale': scale,
                'orig_mean': orig_mean,
                'fused_mean': fused_mean,
                'orig_max': orig_max,
                'fused_max': fused_max,
                'reduction_mean': reduction_mean,
                'reduction_max': reduction_max
            })
    
    # Print results
    print(f"{'Head':<6} {'Dim':<6} {'CRS Scale':<12} {'Original Δ':<14} {'Fused Δ':<14} {'Reduction':<12} {'Status'}")
    print("-" * 90)
    
    for r in results:
        status = "✓ PASS" if r['reduction_mean'] > 1.5 else "✗ FAIL"
        print(f"{r['head']:<6} {r['dim']:<6} {r['scale']:<12.4f} "
              f"{r['orig_mean']:<14.6f} {r['fused_mean']:<14.6f} "
              f"{r['reduction_mean']:<11.2f}x {status}")
    
    print()
    
    # Summary
    if results:
        avg_reduction = np.mean([r['reduction_mean'] for r in results])
        print(f"Summary:")
        print(f"  Outlier dimensions analyzed: {len(results)}")
        print(f"  Average delta reduction: {avg_reduction:.2f}x")
        expected = np.mean([r['scale'] for r in results])
        print(f"  Expected reduction: ~{expected:.2f}x (from CRS scale)")
        print()
        
        if avg_reduction > 1.5:
            print("✓ Weight fusion successfully improved quantization precision!")
        else:
            print("✗ Weight fusion did not improve quantization precision")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Verify Q4_0 block deltas to confirm weight fusion improved precision'
    )
    parser.add_argument('original_gguf', help='Original Q4_0 GGUF file')
    parser.add_argument('fused_gguf', help='CRS-fused Q4_0 GGUF file')
    parser.add_argument('scales_file', help='CRS scales file')
    parser.add_argument('--layer', type=int, default=16, help='Layer to analyze (default: 16)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Q4_0 Block Delta Verification")
    print("="*70)
    
    # Load CRS scales
    print(f"\n[1/3] Loading CRS scales from {args.scales_file}...")
    scales_data = load_crs_scales(args.scales_file)
    print(f"  Loaded scales for {scales_data['n_layers']} layers, {scales_data['n_heads']} heads")
    
    # Construct tensor name
    tensor_name = f"blk.{args.layer}.attn_k.weight"
    
    # Load original deltas
    print(f"\n[2/3] Loading original Q4_0 deltas from {args.original_gguf}...")
    print(f"  Tensor: {tensor_name}")
    orig_deltas, shape = load_q4_0_deltas(args.original_gguf, tensor_name)
    print(f"  Loaded {len(orig_deltas)} blocks")
    
    # Load fused deltas
    print(f"\n[3/3] Loading fused Q4_0 deltas from {args.fused_gguf}...")
    fused_deltas, fused_shape = load_q4_0_deltas(args.fused_gguf, tensor_name)
    print(f"  Loaded {len(fused_deltas)} blocks")
    
    if not np.array_equal(shape, fused_shape):
        print(f"\nError: Shape mismatch! {shape} vs {fused_shape}")
        return 1
    
    # Analyze
    analyze_deltas(orig_deltas, fused_deltas, shape, args.layer, scales_data)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
