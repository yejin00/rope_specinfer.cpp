#!/usr/bin/env python3
import struct
import numpy as np
import sys

MAGIC_APOR = 0x524F5041

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

absmax_data = load_absmax(sys.argv[1])
layer = int(sys.argv[2])
head = int(sys.argv[3])
dim = int(sys.argv[4])

value = absmax_data[layer][head][dim]
print(f"Layer {layer}, Head {head}, Dim {dim}: {value:.4f}")
