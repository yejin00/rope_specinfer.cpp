import struct
import numpy as np
import sys

def check_identity(file_path, layer_idx=0):
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        
        print(f"Header: L={n_layers}, H={n_heads}, D={n_dims}")
        
        # Skip to target layer
        # Each layer has n_heads * (pre + post)
        # Each head has n_dims * 4 bytes for pre, n_dims * 4 bytes for post
        bytes_per_head = n_dims * 4 * 2
        bytes_per_layer = n_heads * bytes_per_head
        
        f.seek(12 + 4 + 4 + layer_idx * bytes_per_layer)
        
        heads_data = []
        for h in range(n_heads):
            pre = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
            post = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
            heads_data.append((pre, post))
            
    # Compare Head 0 with others
    base_pre, base_post = heads_data[0]
    
    print(f"\nChecking Layer {layer_idx} Identity across {n_heads} heads:")
    for h in range(1, n_heads):
        curr_pre, curr_post = heads_data[h]
        if np.array_equal(base_pre, curr_pre):
            print(f"  Head 0 vs Head {h} (Pre):  IDENTICAL ⚠️")
        else:
            diff = np.abs(base_pre - curr_pre).max()
            print(f"  Head 0 vs Head {h} (Pre):  Different (Max Diff: {diff})")
            
        if np.array_equal(base_post, curr_post):
            print(f"  Head 0 vs Head {h} (Post): IDENTICAL ⚠️")
        else:
            diff = np.abs(base_post - curr_post).max()
            print(f"  Head 0 vs Head {h} (Post): Different (Max Diff: {diff})")

if __name__ == "__main__":
    check_identity(sys.argv[1])
