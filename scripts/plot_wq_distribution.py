import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_prs_scales(scales_file):
    with open(scales_file, 'rb') as f:
        magic = f.read(4)
        if magic != b'SRCS':
            raise ValueError(f"Invalid magic: {magic}")
            
        version = struct.unpack('I', f.read(4))[0]
        n_layers = struct.unpack('I', f.read(4))[0]
        n_heads = struct.unpack('I', f.read(4))[0]
        n_dims = struct.unpack('I', f.read(4))[0]
        top_k = struct.unpack('I', f.read(4))[0]
        
        n_outliers = top_k * 2
        scales_map = {}
        
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                indices = np.frombuffer(f.read(n_outliers * 4), dtype=np.int32)
                scales = np.frombuffer(f.read(n_outliers * 4), dtype=np.float32)
                
                for dim, scale in zip(indices, scales):
                    if dim >= 0 and scale > 0:
                        scales_map[(layer_idx, head_idx, int(dim))] = float(scale)
        
        return scales_map, n_layers, n_heads, n_dims

def plot_wq_scaling(scales_map, layer_idx, head_idx, n_dims, output_file):
    # Prepare data
    dims = np.arange(n_dims)
    scales = np.ones(n_dims)
    
    outlier_dims = []
    outlier_scales = []
    
    for dim in dims:
        if (layer_idx, head_idx, dim) in scales_map:
            scale = scales_map[(layer_idx, head_idx, dim)]
            scales[dim] = scale
            outlier_dims.append(dim)
            outlier_scales.append(scale)
            
    # Plot
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"Wq Weight Scaling Factors: L{layer_idx} H{head_idx}", fontsize=16)
    
    # 1. Bar chart of scales per dimension
    ax1 = plt.subplot(2, 1, 1)
    
    # Background blocks
    colors = ['#ffebee', '#fff8e1', '#e8f5e9', '#e8eaf6']
    labels = ['Block 0 (0-32)', 'Block 1 (32-64)', 'Block 2 (64-96)', 'Block 3 (96-128)']
    for i in range(4):
        start = i * 32
        end = min((i + 1) * 32, n_dims)
        ax1.axvspan(start, end, alpha=0.5, color=colors[i], label=labels[i])
        
    ax1.bar(dims, scales, color='gray', alpha=0.5, width=0.8)
    
    if outlier_dims:
        ax1.bar(outlier_dims, outlier_scales, color='red', width=0.8, label='Scaled Dims (PRS)')
        
        # Add text labels
        for dim, scale in zip(outlier_dims, outlier_scales):
            ax1.text(dim, scale + 0.1, f'{scale:.2f}', ha='center', va='bottom', 
                     fontsize=8, rotation=90, bbox=dict(facecolor='white', alpha=0.7, pad=1))
             
    ax1.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel("Scale Factor applied to Wq")
    ax1.set_xlabel("Dimension Index")
    ax1.set_xlim(-1, n_dims)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of applied scales
    ax2 = plt.subplot(2, 2, 3)
    if outlier_scales:
        ax2.hist(outlier_scales, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax2.set_title("Distribution of PRS Scales (> 1.0)")
        ax2.set_xlabel("Scale Factor")
        ax2.set_ylabel("Count")
        
        # Add stats
        stats_text = f"Count: {len(outlier_scales)}\nMin: {min(outlier_scales):.2f}\nMax: {max(outlier_scales):.2f}\nMean: {np.mean(outlier_scales):.2f}"
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, "No scales applied in this head", ha='center', va='center')
        
    # 3. Scale vs Dimension scatter
    ax3 = plt.subplot(2, 2, 4)
    if outlier_scales:
        scatter = ax3.scatter(outlier_dims, outlier_scales, c=outlier_dims, cmap='viridis', s=50, edgecolor='black')
        ax3.set_title("Scale Factor by Dimension")
        ax3.set_xlabel("Dimension Index")
        ax3.set_ylabel("Scale Factor")
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Dimension')
    else:
        ax3.text(0.5, 0.5, "No scales applied in this head", ha='center', va='center')
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Wq distribution plot to {output_file}")
    
    # Print summary
    print(f"\nLayer {layer_idx}, Head {head_idx} Summary:")
    print(f"Total dimensions scaled: {len(outlier_dims)} / {n_dims}")
    if outlier_scales:
        print(f"Max scale applied to Wq: {max(outlier_scales):.4f}")
        print(f"Average scale applied:   {np.mean(outlier_scales):.4f}")
        print(f"Dimensions with scale > 5.0: {sum(1 for s in outlier_scales if s > 5.0)}")
        print(f"Dimensions with scale > 10.0: {sum(1 for s in outlier_scales if s > 10.0)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scales_file', help='PRS scales binary file')
    parser.add_argument('--layer', type=int, default=29)
    parser.add_argument('--head', type=int, default=6)
    parser.add_argument('--output', default='wq_distribution.png')
    
    args = parser.parse_args()
    
    print(f"Loading scales from {args.scales_file}...")
    scales_map, n_layers, n_heads, n_dims = load_prs_scales(args.scales_file)
    
    plot_wq_scaling(scales_map, args.layer, args.head, n_dims, args.output)

if __name__ == '__main__':
    main()
