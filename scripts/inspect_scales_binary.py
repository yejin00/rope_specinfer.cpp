import struct
import numpy as np
import sys
import os

def inspect(path):
    if not os.path.exists(path):
        print(f"Error: File {path} not found.")
        return

    with open(path, 'rb') as f:
        # Header
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        top_k = struct.unpack('I', f.read(4))[0]
        
        print(f"File: {path}")
        print(f"Magic: {hex(magic)}")
        print(f"Version: {version}")
        print(f"Layers: {n_layers}")
        print(f"Heads: {n_heads}")
        print(f"Dims: {n_dims}")
        print(f"Top-k (pairs in header): {top_k}")
        
        # Determine stored length
        header_size = 4 * 6
        current_pos = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(current_pos)
        
        remaining = file_size - header_size
        block_size_crs = n_layers * n_heads * (top_k * 4 + top_k * 4)
        block_size_prs = n_layers * n_heads * (top_k * 2 * 4 + top_k * 2 * 4)
        
        if remaining == block_size_prs:
            print("Format: PRS (2 * top_k per head)")
            stored_k = top_k * 2
        elif remaining == block_size_crs:
            print("Format: CRS (top_k per head)")
            stored_k = top_k
        else:
            print(f"Warning: Unknown size. Remaining: {remaining}, Expected PRS: {block_size_prs}")
            stored_k = top_k * 2

        print("-" * 40)
        
        # Read Data
        all_scales = []
        
        for l in range(n_layers):
            for h in range(n_heads):
                indices = np.frombuffer(f.read(stored_k * 4), dtype=np.int32)
                scales = np.frombuffer(f.read(stored_k * 4), dtype=np.float32)
                
                # Filter invalid (-1)
                valid_mask = indices != -1
                valid_indices = indices[valid_mask]
                valid_scales = scales[valid_mask]
                
                all_scales.extend(valid_scales)
                
                if l == 0 and h == 0:
                    print(f"Sample (Layer 0, Head 0):")
                    for i, s in zip(valid_indices, valid_scales):
                        print(f"  Dim {i}: Scale {s:.4f}")
        
        all_scales = np.array(all_scales)
        print("-" * 40)
        print(f"Total valid scales: {len(all_scales)}")
        if len(all_scales) > 0:
            print(f"Min Scale: {all_scales.min():.4f}")
            print(f"Max Scale: {all_scales.max():.4f}")
            print(f"Mean Scale: {all_scales.mean():.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect.py <bin_file>")
    else:
        inspect(sys.argv[1])
