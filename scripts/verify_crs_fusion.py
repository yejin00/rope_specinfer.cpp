#!/usr/bin/env python3
"""
Verify CRS weight fusion by comparing original and fused GGUF weights.

Checks:
1. Only specified outlier channels are modified
2. WK is divided by scale (K becomes smaller)
3. WQ is multiplied by scale (Q becomes larger)
4. QK^T invariance is preserved
"""

import struct
import numpy as np
import argparse
import sys

try:
    import gguf
except ImportError:
    print("Error: gguf library not found. Install with: pip install gguf")
    sys.exit(1)


MAGIC_SCRS = 0x53435253  # 'SCRS'


def load_crs_scales(scales_path):
    """Load static CRS scales"""
    with open(scales_path, 'rb') as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != MAGIC_SCRS:
            raise ValueError(f"Invalid magic number: {hex(magic)}")
        
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        top_k = struct.unpack('I', f.read(4))[0]
        
        scales_data = []
        for layer_idx in range(n_layers):
            layer_scales = []
            for head_idx in range(n_heads):
                indices = np.frombuffer(f.read(top_k * 4), dtype=np.int32)
                scales = np.frombuffer(f.read(top_k * 4), dtype=np.float32)
                
                valid_mask = indices >= 0
                valid_indices = indices[valid_mask]
                valid_scales = scales[valid_mask]
                
                layer_scales.append({
                    'indices': valid_indices,
                    'scales': valid_scales
                })
            scales_data.append(layer_scales)
        
        return {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_dims': n_dims,
            'scales': scales_data
        }


def load_gguf_weights(gguf_path):
    """Load Q and K weights from GGUF"""
    reader = gguf.GGUFReader(gguf_path)
    
    weights = {}
    for tensor in reader.tensors:
        if '.attn_q.weight' in tensor.name or '.attn_k.weight' in tensor.name:
            parts = tensor.name.split('.')
            if len(parts) > 1 and parts[0] == 'blk':
                layer_idx = int(parts[1])
                weight_type = 'q' if 'attn_q' in tensor.name else 'k'
                
                # Convert to float32
                weight = tensor.data.astype(np.float32)
                weights[(layer_idx, weight_type)] = weight
    
    return weights


def verify_weight_modification(original_wk, fused_wk, original_wq, fused_wq, 
                                layer_idx, scales_data, tolerance=1e-3):
    """
    Verify that weights are modified correctly:
    - WK_fused[:, outlier_ch] = WK_original[:, outlier_ch] / scale
    - WQ_fused[:, outlier_ch] = WQ_original[:, outlier_ch] * scale
    """
    n_heads = scales_data['n_heads']
    n_dims = scales_data['n_dims']
    layer_scales = scales_data['scales'][layer_idx]
    
    results = {
        'modified_channels': [],
        'wk_correct': [],
        'wq_correct': [],
        'unmodified_correct': True,
        'details': []  # Detailed info per modification
    }
    
    # Check each head
    for head_idx, head_data in enumerate(layer_scales):
        indices = head_data['indices']
        scales = head_data['scales']
        
        if len(indices) == 0:
            continue
        
        base_offset = head_idx * n_dims
        
        for idx, scale in zip(indices, scales):
            col_idx = base_offset + idx
            results['modified_channels'].append(col_idx)
            
            # Check WK: should be divided by scale
            wk_original = original_wk[:, col_idx]
            wk_fused = fused_wk[:, col_idx]
            wk_expected = wk_original / scale
            wk_diff = np.abs(wk_fused - wk_expected).max()
            wk_pass = wk_diff < tolerance
            results['wk_correct'].append(wk_pass)
            
            # Statistics for WK
            wk_orig_median = np.median(np.abs(wk_original))
            wk_orig_max = np.max(np.abs(wk_original))
            wk_fused_median = np.median(np.abs(wk_fused))
            wk_fused_max = np.max(np.abs(wk_fused))
            
            # Check WQ: should be multiplied by scale
            # For GQA: need to handle multiple Q heads per KV head
            n_heads_q = original_wq.shape[1] // n_dims
            n_heads_kv = n_heads
            num_q_groups = n_heads_q // n_heads_kv
            
            wq_details = []
            for group_idx in range(num_q_groups):
                q_head_idx = head_idx * num_q_groups + group_idx
                q_col_idx = q_head_idx * n_dims + idx
                
                if q_col_idx < original_wq.shape[1]:
                    wq_original = original_wq[:, q_col_idx]
                    wq_fused = fused_wq[:, q_col_idx]
                    wq_expected = wq_original * scale
                    wq_diff = np.abs(wq_fused - wq_expected).max()
                    wq_pass = wq_diff < tolerance
                    results['wq_correct'].append(wq_pass)
                    
                    wq_orig_median = np.median(np.abs(wq_original))
                    wq_fused_median = np.median(np.abs(wq_fused))
                    
                    wq_details.append({
                        'q_head': q_head_idx,
                        'pass': wq_pass,
                        'orig_median': wq_orig_median,
                        'fused_median': wq_fused_median
                    })
            
            # Store detailed info
            results['details'].append({
                'head': head_idx,
                'dim': idx,
                'global_col': col_idx,
                'scale': scale,
                'wk': {
                    'pass': wk_pass,
                    'diff': wk_diff,
                    'orig_median': wk_orig_median,
                    'orig_max': wk_orig_max,
                    'fused_median': wk_fused_median,
                    'fused_max': wk_fused_max
                },
                'wq': wq_details
            })
    
    # Check that unmodified channels are unchanged
    all_cols = set(range(original_wk.shape[1]))
    modified_cols = set(results['modified_channels'])
    unmodified_cols = list(all_cols - modified_cols)
    
    if len(unmodified_cols) > 0:
        sample_cols = unmodified_cols[:min(10, len(unmodified_cols))]
        for col_idx in sample_cols:
            wk_diff = np.abs(original_wk[:, col_idx] - fused_wk[:, col_idx]).max()
            if wk_diff > tolerance:
                results['unmodified_correct'] = False
                break
    
    return results


def verify_qk_invariance(original_wq, fused_wq, original_wk, fused_wk, 
                         layer_idx, scales_data, tolerance=1e-2):
    """
    Verify QK^T invariance:
    Q_fused @ K_fused^T ≈ Q_original @ K_original^T
    """
    # Sample a few rows for efficiency
    n_samples = min(100, original_wq.shape[0])
    sample_rows = np.random.choice(original_wq.shape[0], n_samples, replace=False)
    
    # Compute QK^T for sampled rows
    qk_original = original_wq[sample_rows, :] @ original_wk.T
    qk_fused = fused_wq[sample_rows, :] @ fused_wk.T
    
    # Check difference
    diff = np.abs(qk_original - qk_fused)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'invariant': max_diff < tolerance
    }


def main():
    parser = argparse.ArgumentParser(description='Verify CRS weight fusion')
    parser.add_argument('original_gguf', help='Original GGUF file')
    parser.add_argument('fused_gguf', help='Fused GGUF file')
    parser.add_argument('scales_file', help='CRS scales file')
    parser.add_argument('--layers', type=int, nargs='+', help='Layers to check (default: 0,15,31)')
    parser.add_argument('--tolerance', type=float, default=1e-3, help='Numerical tolerance')
    
    args = parser.parse_args()
    
    print("Loading CRS scales...")
    scales_data = load_crs_scales(args.scales_file)
    
    print(f"Loading original weights from: {args.original_gguf}")
    original_weights = load_gguf_weights(args.original_gguf)
    
    print(f"Loading fused weights from: {args.fused_gguf}")
    fused_weights = load_gguf_weights(args.fused_gguf)
    
    # Determine which layers to check
    if args.layers:
        layers_to_check = args.layers
    else:
        layers_to_check = [0, 15, 31]
    
    print(f"\n{'='*60}")
    print(f"Verification Results")
    print(f"{'='*60}\n")
    
    all_passed = True
    
    for layer_idx in layers_to_check:
        print(f"\n{'='*70}")
        print(f"Layer {layer_idx}")
        print(f"{'='*70}")
        
        # Get weights
        original_wk = original_weights.get((layer_idx, 'k'))
        fused_wk = fused_weights.get((layer_idx, 'k'))
        original_wq = original_weights.get((layer_idx, 'q'))
        fused_wq = fused_weights.get((layer_idx, 'q'))
        
        if original_wk is None or fused_wk is None:
            print(f"  ⚠ Missing weights for layer {layer_idx}")
            continue
        
        # Verify weight modification
        mod_results = verify_weight_modification(
            original_wk, fused_wk, original_wq, fused_wq,
            layer_idx, scales_data, args.tolerance
        )
        
        n_modified = len(mod_results['modified_channels'])
        wk_pass = all(mod_results['wk_correct'])
        wq_pass = all(mod_results['wq_correct'])
        unmod_pass = mod_results['unmodified_correct']
        
        print(f"\n📊 Summary:")
        print(f"  Modified channels: {n_modified}")
        print(f"  WK division by scale: {'✓ PASS' if wk_pass else '✗ FAIL'}")
        print(f"  WQ multiplication by scale: {'✓ PASS' if wq_pass else '✗ FAIL'}")
        print(f"  Unmodified channels unchanged: {'✓ PASS' if unmod_pass else '✗ FAIL'}")
        
        # Print detailed info for each modified channel
        print(f"\n📝 Detailed Channel Modifications:")
        for detail in mod_results['details']:
            print(f"\n  🔹 Head {detail['head']}, Dim {detail['dim']} (Global Col {detail['global_col']})")
            print(f"     Scale: {detail['scale']:.6f}")
            
            wk = detail['wk']
            print(f"     WK:")
            print(f"       Original - median: {wk['orig_median']:.6f}, max: {wk['orig_max']:.6f}")
            print(f"       Fused    - median: {wk['fused_median']:.6f}, max: {wk['fused_max']:.6f}")
            print(f"       Ratio (fused/orig): median={wk['fused_median']/wk['orig_median']:.6f}, max={wk['fused_max']/wk['orig_max']:.6f}")
            print(f"       Expected ratio: {1/detail['scale']:.6f}")
            print(f"       Verification: {'✓ PASS' if wk['pass'] else '✗ FAIL'} (diff={wk['diff']:.6f})")
            
            print(f"     WQ ({len(detail['wq'])} Q heads):")
            for wq in detail['wq']:
                print(f"       Q Head {wq['q_head']}: orig_median={wq['orig_median']:.6f}, fused_median={wq['fused_median']:.6f}, ratio={wq['fused_median']/wq['orig_median']:.6f}, expected={detail['scale']:.6f} {'✓' if wq['pass'] else '✗'}")
        
        # Verify QK^T invariance
        qk_results = verify_qk_invariance(
            original_wq, fused_wq, original_wk, fused_wk,
            layer_idx, scales_data, args.tolerance
        )
        
        print(f"\n🔍 QK^T Invariance Check:")
        print(f"  Status: {'✓ PASS' if qk_results['invariant'] else '✗ FAIL (within FP16 precision)'}")
        print(f"  Max diff: {qk_results['max_diff']:.6f}")
        print(f"  Mean diff: {qk_results['mean_diff']:.6f}")
        
        layer_passed = wk_pass and wq_pass and unmod_pass and qk_results['invariant']
        all_passed = all_passed and layer_passed
    
    print(f"{'='*60}")
    if all_passed:
        print("✓ All checks PASSED!")
    else:
        print("✗ Some checks FAILED")
    print(f"{'='*60}")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
