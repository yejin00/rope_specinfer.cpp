import struct
import numpy as np
import sys

def check_ropv_identity(file_path):
    print(f"Checking {file_path}...")
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != 0x524F5056: # ROPV
            print(f"Not a ROPV file (Magic: {hex(magic)})")
            return

        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        n_tokens = struct.unpack('I', f.read(4))[0]
        
        print(f"Header: L={n_layers}, H={n_heads}, D={n_dims}, T={n_tokens}")
        
        # Check Layer 0
        layer = 0
        # Pre
        pre_count = struct.unpack('I', f.read(4))[0]
        print(f"Layer {layer} Pre count: {pre_count}")
        if pre_count > 0:
            pre_data = np.frombuffer(f.read(pre_count*4), dtype=np.float32)
            # Try to split into heads assuming [H, T, D] layout for verification
            # If [T, H, D], splitting into H chunks means splitting by time.
            chunk_size = pre_count // n_heads
            chunks = [pre_data[i*chunk_size : (i+1)*chunk_size] for i in range(n_heads)]
            
            print("  Pre-RoPE Chunks (assuming [H, T, D]):")
            for h in range(1, n_heads):
                if np.array_equal(chunks[0], chunks[h]):
                    print(f"    Chunk 0 == Chunk {h} (IDENTICAL)")
                else:
                    diff = np.abs(chunks[0] - chunks[h]).max()
                    print(f"    Chunk 0 != Chunk {h} (Diff: {diff:.4f})")
        
        # Post
        post_count = struct.unpack('I', f.read(4))[0]
        print(f"Layer {layer} Post count: {post_count}")
        if post_count > 0:
            post_data = np.frombuffer(f.read(post_count*4), dtype=np.float32)
            chunk_size = post_count // n_heads
            chunks = [post_data[i*chunk_size : (i+1)*chunk_size] for i in range(n_heads)]
            
            print("  Post-RoPE Chunks (assuming [H, T, D]):")
            for h in range(1, n_heads):
                if np.array_equal(chunks[0], chunks[h]):
                    print(f"    Chunk 0 == Chunk {h} (IDENTICAL ⚠️)")
                else:
                    diff = np.abs(chunks[0] - chunks[h]).max()
                    print(f"    Chunk 0 != Chunk {h} (Diff: {diff:.4f})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_ropv_identity(sys.argv[1])
    else:
        print("Usage: python debug_ropv_identity.py <file>")
