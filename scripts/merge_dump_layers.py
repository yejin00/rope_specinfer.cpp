#!/usr/bin/env python3
"""
Merge separate per-layer dump files into a single ROPV binary file.
"""

import struct
import argparse
import os
import glob
import re

MAGIC_ROPV = 0x524F5056  # "ROPV"

def parse_layer_idx(filename, prefix):
    # filename format: {prefix}_layer_{idx}.bin
    base = os.path.basename(filename)
    prefix_base = os.path.basename(prefix)
    
    # Simple regex
    m = re.search(r'layer_(\d+)\.bin$', base)
    if m:
        return int(m.group(1))
    return -1

def main():
    parser = argparse.ArgumentParser(description='Merge per-layer dump files into ROPV format')
    parser.add_argument('--prefix', required=True, help='Prefix used for dumping (e.g. dump/act)')
    parser.add_argument('--output', required=True, help='Output .bin file')
    parser.add_argument('--heads', type=int, default=8, help='Number of KV heads')
    parser.add_argument('--dims', type=int, default=128, help='Head dimension')
    parser.add_argument('--total-layers', type=int, default=32, help='Total layers')
    
    args = parser.parse_args()
    
    prefix = args.prefix
    # Find files
    # We expect: prefix_layer_0.bin, prefix_layer_1.bin ...
    # Construct glob pattern
    # Use dirname + basename
    dir_name = os.path.dirname(prefix)
    base_name = os.path.basename(prefix)
    
    if dir_name == '': dir_name = '.'
    
    pattern = f"{dir_name}/{base_name}_layer_*.bin"
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching {pattern}")
        return
    
    # Sort files by layer index
    layer_files = {}
    for f in files:
        idx = parse_layer_idx(f, prefix)
        if idx >= 0:
            layer_files[idx] = f
            
    print(f"Found {len(layer_files)} layer files.")
    
    if len(layer_files) == 0:
        return

    # Determine total tokens from the first file
    first_layer_file = layer_files[min(layer_files.keys())]
    file_size = os.path.getsize(first_layer_file)
    bytes_per_token = args.heads * args.dims * 4 # float32
    total_tokens = file_size // bytes_per_token
    
    print(f"Estimated tokens: {total_tokens} (based on {file_size} bytes / {bytes_per_token} per token)")
    
    # Write ROPV file
    with open(args.output, 'wb') as fout:
        # Header
        fout.write(struct.pack('I', MAGIC_ROPV))
        fout.write(struct.pack('I', 1)) # Version
        fout.write(struct.pack('I', args.total_layers))
        fout.write(struct.pack('I', args.heads))
        fout.write(struct.pack('I', args.dims))
        fout.write(struct.pack('I', total_tokens))
        
        for layer in range(args.total_layers):
            if layer in layer_files:
                fpath = layer_files[layer]
                with open(fpath, 'rb') as fin:
                    data = fin.read()
                    
                # Write as PRE values (count + data)
                # Count is number of float elements
                count = len(data) // 4
                fout.write(struct.pack('I', count))
                fout.write(data)
                
                # Write POST values (0 count) - We only dumped one stage
                fout.write(struct.pack('I', 0))
                
                print(f"Merged Layer {layer}: {count} elements")
            else:
                print(f"Warning: Missing file for Layer {layer}, writing empty.")
                fout.write(struct.pack('I', 0)) # Pre count 0
                fout.write(struct.pack('I', 0)) # Post count 0

    print(f"\nSuccessfully merged into {args.output}")

if __name__ == '__main__':
    main()
