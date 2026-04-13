#!/usr/bin/env python3
"""
Apply FireQ late CRS scales from a .scrs file to the post-RoPE values in a
ROPV dump and write a new ROPV file.

This simulates the runtime K-side late CRS path offline:
    K_post_crs[idx] = K_post_rope[idx] * (1 / q_scale[idx])

The input file is never modified in place.
"""

import argparse
import struct

import numpy as np


MAGIC_ROPV = 0x524F5056  # "ROPV"
MAGIC_SCRS = 0x53435253  # "SCRS"


def read_exact_f32(f, n_floats):
    raw = f.read(n_floats * 4)
    got = len(raw) // 4
    if got < n_floats:
        raise ValueError(f"Expected {n_floats} float32 values, got {got}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def read_exact_i32(f, n_ints):
    raw = f.read(n_ints * 4)
    got = len(raw) // 4
    if got < n_ints:
        raise ValueError(f"Expected {n_ints} int32 values, got {got}")
    return np.frombuffer(raw, dtype=np.int32).copy()


def load_ropv_full(path):
    with open(path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_ROPV:
            raise ValueError(f"Invalid ROPV magic: {hex(magic)}")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]
        n_tokens = struct.unpack("I", f.read(4))[0]
        stride = n_heads * n_dims

        print(f"Input ROPV: {path}")
        print(f"  version={version}, layers={n_layers}, heads={n_heads}, dims={n_dims}, tokens={n_tokens}")

        layers = []
        for layer_idx in range(n_layers):
            pre_count = struct.unpack("I", f.read(4))[0]
            if pre_count > 0:
                pre = read_exact_f32(f, pre_count)
                pre = pre.reshape(pre_count // stride, n_heads, n_dims)
            else:
                pre = None

            post_count = struct.unpack("I", f.read(4))[0]
            if post_count > 0:
                post = read_exact_f32(f, post_count)
                post = post.reshape(post_count // stride, n_heads, n_dims)
            else:
                post = None

            layers.append({
                "pre": pre,
                "post": post,
            })

            if layer_idx == 0:
                pre_tok = pre.shape[0] if pre is not None else 0
                post_tok = post.shape[0] if post is not None else 0
                print(
                    f"  Layer 0: pre_count={pre_count} -> {pre_tok} tokens, "
                    f"post_count={post_count} -> {post_tok} tokens"
                )

        return {
            "version": version,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_dims": n_dims,
            "n_tokens": n_tokens,
            "layers": layers,
        }


def load_scrs(path):
    with open(path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        if magic != MAGIC_SCRS:
            raise ValueError(f"Invalid SCRS magic: {hex(magic)}")

        version = struct.unpack("I", f.read(4))[0]
        n_layers = struct.unpack("I", f.read(4))[0]
        n_heads = struct.unpack("I", f.read(4))[0]
        n_dims = struct.unpack("I", f.read(4))[0]
        top_k = struct.unpack("I", f.read(4))[0]

        print(f"Input SCRS: {path}")
        print(f"  version={version}, layers={n_layers}, heads={n_heads}, dims={n_dims}, top_k={top_k}")

        data = []
        for _layer in range(n_layers):
            layer = []
            for _head in range(n_heads):
                indices = read_exact_i32(f, top_k)
                scales = read_exact_f32(f, top_k)
                layer.append({
                    "indices": indices,
                    "scales": scales,
                })
            data.append(layer)

        return {
            "version": version,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_dims": n_dims,
            "top_k": top_k,
            "data": data,
        }


def apply_k_side_scrs(ropv, scrs):
    if ropv["n_layers"] != scrs["n_layers"]:
        raise ValueError(
            f"Layer count mismatch: ROPV={ropv['n_layers']} vs SCRS={scrs['n_layers']}"
        )
    if ropv["n_heads"] != scrs["n_heads"]:
        raise ValueError(
            f"Head count mismatch: ROPV={ropv['n_heads']} vs SCRS={scrs['n_heads']}"
        )
    if ropv["n_dims"] != scrs["n_dims"]:
        raise ValueError(
            f"Dim count mismatch: ROPV={ropv['n_dims']} vs SCRS={scrs['n_dims']}"
        )

    touched_dims = 0
    touched_token_instances = 0

    for layer_idx in range(ropv["n_layers"]):
        post = ropv["layers"][layer_idx]["post"]
        if post is None:
            continue

        for head_idx in range(ropv["n_heads"]):
            head_data = scrs["data"][layer_idx][head_idx]
            for idx, scale in zip(head_data["indices"], head_data["scales"]):
                idx = int(idx)
                scale = float(scale)
                if idx < 0 or idx >= ropv["n_dims"] or scale <= 1e-6:
                    continue

                # Runtime K path uses reciprocal scales.
                post[:, head_idx, idx] *= np.float32(1.0 / scale)
                touched_dims += 1
                touched_token_instances += post.shape[0]

    return touched_dims, touched_token_instances


def write_ropv(path, data):
    with open(path, "wb") as f:
        f.write(struct.pack("I", MAGIC_ROPV))
        f.write(struct.pack("I", data["version"]))
        f.write(struct.pack("I", data["n_layers"]))
        f.write(struct.pack("I", data["n_heads"]))
        f.write(struct.pack("I", data["n_dims"]))
        f.write(struct.pack("I", data["n_tokens"]))

        for layer in data["layers"]:
            pre = layer["pre"]
            if pre is not None:
                f.write(struct.pack("I", pre.size))
                f.write(pre.astype(np.float32).tobytes())
            else:
                f.write(struct.pack("I", 0))

            post = layer["post"]
            if post is not None:
                f.write(struct.pack("I", post.size))
                f.write(post.astype(np.float32).tobytes())
            else:
                f.write(struct.pack("I", 0))


def main():
    parser = argparse.ArgumentParser(
        description="Apply FireQ late CRS (.scrs) to post-RoPE values in a ROPV dump"
    )
    parser.add_argument("input_ropv", help="Input ROPV file (post-RoPE before CRS)")
    parser.add_argument("input_scrs", help="Input FireQ late CRS .scrs file")
    parser.add_argument("output_ropv", help="Output ROPV file (post-RoPE after simulated CRS)")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report, but do not write output")
    args = parser.parse_args()

    ropv = load_ropv_full(args.input_ropv)
    scrs = load_scrs(args.input_scrs)

    if args.dry_run:
        print("\nDry run requested. No output file written.")
        return

    touched_dims, touched_token_instances = apply_k_side_scrs(ropv, scrs)
    print(f"\nApplied K-side SCRS entries: {touched_dims}")
    print(f"Touched token instances:     {touched_token_instances}")

    write_ropv(args.output_ropv, ropv)
    print(f"\nSaved simulated post-CRS ROPV: {args.output_ropv}")


if __name__ == "__main__":
    main()
