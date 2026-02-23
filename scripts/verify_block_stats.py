
import argparse
import numpy as np
import struct
import sys

def read_apor(path):
    with open(path, 'rb') as f:
        # Header
        magic, version, n_layers, n_heads, n_dims = struct.unpack('IIIII', f.read(20))
        # Skip tokens count (another 4 bytes) - wait, standard APOR might not have tokens?
        # Let's check extract_absmax_from_full.py writer.
        # It writes: MAGIC, version, n_layers, n_heads, n_dims. 
        # Wait, the reader in block_extract... says:
        # magic(4), version(4), n_layers(4), n_heads(4), n_dims(4).
        # But verify_block_stats need to be robust.
        pass
    
    # Let's use the same reader as block_extract_crs_from_absmax.py
    with open(path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        
        data = np.zeros((n_layers, n_heads, n_dims), dtype=np.float32)
        
        for l in range(n_layers):
            for h in range(n_heads):
                pre = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                post = np.frombuffer(f.read(n_dims * 4), dtype=np.float32)
                data[l, h] = pre # We use pre
                
    return data, n_dims

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('orig_file', help='Original AbsMax file')
    parser.add_argument('fused_file', help='Fused AbsMax file')
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--head', type=int, required=True)
    parser.add_argument('--block-size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max-scale', type=float, default=2.0)
    args = parser.parse_args()

    print(f"Loading {args.orig_file}...")
    orig_data, n_dims = read_apor(args.orig_file)
    print(f"Loading {args.fused_file}...")
    fused_data, _ = read_apor(args.fused_file)

    l, h = args.layer, args.head
    orig = orig_data[l, h]
    fused = fused_data[l, h]

    print(f"\n=== Analysis: Layer {l}, Head {h} ===")
    
    # Iterate over blocks
    n_blocks = (n_dims + args.block_size - 1) // args.block_size
    
    for b in range(n_blocks):
        start = b * args.block_size
        end = min((b + 1) * args.block_size, n_dims)
        
        block_orig = orig[start:end]
        block_fused = fused[start:end]
        
        # Identify outliers (where orig != fused significantly)
        diff = np.abs(block_orig - block_fused)
        outlier_mask = diff > 0.01  # simple threshold
        
        # Calculate block median from non-outliers in ORIGINAL
        # (Since we want to verify the logic used during extraction)
        # Note: In extraction, we selected Top-K pairs. 
        # Here we just check the median of what wasn't changed? 
        # Or better: Median of the FUSED non-outliers (which should be same as Orig non-outliers)
        
        non_outlier_vals = block_orig[~outlier_mask]
        if len(non_outlier_vals) > 0:
            median = np.median(non_outlier_vals)
        else:
            median = 0.0
            
        print(f"\n[Block {b}: Dims {start}-{end-1}] Median (Non-Outliers): {median:.4f}")
        
        outlier_indices = np.where(outlier_mask)[0]
        if len(outlier_indices) > 0:
            print(f"  Found {len(outlier_indices)} outliers:")
            for idx in outlier_indices:
                abs_idx = start + idx
                val_orig = block_orig[idx]
                val_fused = block_fused[idx]
                
                # Expected logic:
                # scale = (val_orig / median) ^ alpha
                # scale = min(scale, max_scale)
                # expected_val = val_orig / scale
                
                raw_scale = val_orig / median if median > 1e-6 else 1.0
                damped_scale = max(raw_scale, 1.0) ** args.alpha
                final_scale = min(damped_scale, args.max_scale)
                final_scale = max(1.0, final_scale)
                
                expected_val = val_orig / final_scale
                
                error = abs(val_fused - expected_val)
                match = "MATCH" if error < 0.1 else "MISMATCH"
                
                print(f"    Dim {abs_idx}: Orig={val_orig:.4f} -> Fused={val_fused:.4f} | Target={expected_val:.4f} (Scale={final_scale:.3f}) [{match}]")
        else:
            print("  No outliers modified.")

if __name__ == '__main__':
    main()
