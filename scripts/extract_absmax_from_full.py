#!/usr/bin/env python3
import argparse
import struct

import numpy as np


MAGIC_ROPV = 0x524F5056
MAGIC_APOR = 0x524F5041


def main():
    parser = argparse.ArgumentParser(description='Extract absmax from ROPV full dump')
    parser.add_argument('input_ropv', help='Input ROPV file (rope_values_*.bin)')
    parser.add_argument('output_apor', help='Output APOR file (rope_absmax_*.bin)')
    args = parser.parse_args()

    print(f'Reading {args.input_ropv} ...')
    with open(args.input_ropv, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f'Invalid magic: {hex(magic)}, expected ROPV')

        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        n_tokens = struct.unpack('I', f.read(4))[0]

        print(f'  layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}')

        pre_absmax = []
        post_absmax = []

        for layer in range(n_layers):
            pre_count = struct.unpack('I', f.read(4))[0]
            if pre_count > 0:
                pre_data = np.frombuffer(f.read(pre_count * 4), dtype=np.float32)
                pre_data = pre_data.reshape(-1, n_heads, n_dims)
                layer_pre_absmax = np.max(np.abs(pre_data), axis=0)
            else:
                layer_pre_absmax = np.zeros((n_heads, n_dims), dtype=np.float32)

            post_count = struct.unpack('I', f.read(4))[0]
            if post_count > 0:
                post_data = np.frombuffer(f.read(post_count * 4), dtype=np.float32)
                post_data = post_data.reshape(-1, n_heads, n_dims)
                layer_post_absmax = np.max(np.abs(post_data), axis=0)
            else:
                layer_post_absmax = np.zeros((n_heads, n_dims), dtype=np.float32)

            pre_absmax.append(layer_pre_absmax)
            post_absmax.append(layer_post_absmax)

            if (layer + 1) % 8 == 0:
                print(f'  Processed {layer + 1}/{n_layers} layers...')

    print(f'Writing {args.output_apor} ...')
    with open(args.output_apor, 'wb') as f:
        f.write(struct.pack('I', MAGIC_APOR))
        f.write(struct.pack('I', 1))
        f.write(struct.pack('I', n_layers))
        f.write(struct.pack('I', n_heads))
        f.write(struct.pack('I', n_dims))

        for layer in range(n_layers):
            for head in range(n_heads):
                f.write(pre_absmax[layer][head].astype(np.float32).tobytes())
                f.write(post_absmax[layer][head].astype(np.float32).tobytes())

    print(f'✓ Done: {args.output_apor}')


if __name__ == '__main__':
    main()
