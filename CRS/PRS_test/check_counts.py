import struct
import sys

def check(file_path):
    with open(file_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        n_tokens = struct.unpack('I', f.read(4))[0]
        
        print(f"Layers: {n_layers}, Heads: {n_heads}, Dims: {n_dims}, Tokens: {n_tokens}")
        
        for l in range(n_layers):
            pre_count = struct.unpack('I', f.read(4))[0]
            f.seek(pre_count * 4, 1)
            post_count = struct.unpack('I', f.read(4))[0]
            f.seek(post_count * 4, 1)
            print(f"Layer {l}: pre_count={pre_count}, post_count={post_count}")

check(sys.argv[1])
