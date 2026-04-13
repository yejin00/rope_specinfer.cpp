#!/usr/bin/env python3
import struct
import numpy as np
import sys

MAGIC_APOR = 0x524F5041
MAGIC_SCRS = 0x53435253

def load_absmax(file_path):
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_APOR:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        
        pre_absmax = []
        
        for layer in range(n_layers):
            layer_pre = []
            for head in range(n_heads):
                data_pre = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                layer_pre.append(data_pre)
                data_post = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
            
            pre_absmax.append(np.array(layer_pre))
            
        return pre_absmax

def load_outliers(scales_file, layer, head):
    with open(scales_file, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_SCRS:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        top_k = struct.unpack('I', f.read(4))[0]

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
        
        # Skip to target layer/head
        bytes_per_head = stored_len * 4 + stored_len * 4
        skip_bytes = (layer * n_heads + head) * bytes_per_head
        f.seek(header_bytes + skip_bytes, 0)
        
        indices = np.frombuffer(f.read(stored_len * 4), dtype=np.int32)
        valid_mask = indices >= 0
        return set(indices[valid_mask])

orig_data = load_absmax(sys.argv[1])
prs_data = load_absmax(sys.argv[2])
layer = int(sys.argv[3])
head = int(sys.argv[4])

outlier_set = load_outliers(sys.argv[5], layer, head)

print(f"\nBlock 2 (64-96) values for Layer {layer}, Head {head}:")
print(f"{'Dim':<6} {'Original':<12} {'PRS-Fused':<12} {'Outlier':<10}")
print("-" * 50)

for dim in range(64, 96):
    orig_val = orig_data[layer][head][dim]
    prs_val = prs_data[layer][head][dim]
    is_outlier = "YES" if dim in outlier_set else "NO"
    print(f"{dim:<6} {orig_val:<12.4f} {prs_val:<12.4f} {is_outlier:<10}")

# Find max non-outlier in block 2
max_val = 0
max_dim = -1
for dim in range(64, 96):
    if dim not in outlier_set:
        val = prs_data[layer][head][dim]
        if val > max_val:
            max_val = val
            max_dim = dim

print(f"\nMax non-outlier in Block 2: dim {max_dim} = {max_val:.4f}")
