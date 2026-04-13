import argparse
import struct
import numpy as np
import matplotlib.pyplot as plt
import os
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
        
        return scales_map, n_layers, n_heads

def read_layer_dump(layer_file, n_heads=8, n_dims=128):
    file_size = os.path.getsize(layer_file)
    n_elements = file_size // 4
    elems_per_token = n_heads * n_dims
    n_tokens = n_elements // elems_per_token
    
    data = np.fromfile(layer_file, dtype=np.float32)
    try:
        data = data.reshape(n_tokens, n_heads, n_dims)
        return data, n_tokens
    except ValueError:
        return None, 0

def calculate_ratios_for_block(head_data, start, end, scales_map, layer_idx, head_idx):
    n_tokens = head_data.shape[0]
    block_data = head_data[:, start:end]
    
    ratios = np.zeros(n_tokens)
    m_origs = np.zeros(n_tokens)
    
    # Pre-fetch scales for this block
    block_scales = np.ones(end - start)
    for i in range(end - start):
        block_scales[i] = scales_map.get((layer_idx, head_idx, start + i), 1.0)
        
    for i in range(n_tokens):
        token_data = block_data[i]
        m_orig = np.max(np.abs(token_data))
        m_origs[i] = m_orig
        
        if m_orig < 1e-8:
            ratios[i] = 1.0
            continue
            
        scaled_data = np.abs(token_data) / block_scales
        m_scaled = np.max(scaled_data)
        ratios[i] = m_scaled / m_orig if m_orig > 0 else 1.0
        
    return ratios, m_origs

def plot_analysis(ratios, m_origs, layer_idx, head_idx, block_idx, start, end, output_prefix):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"Step Reduction Analysis: L{layer_idx} H{head_idx} Block {block_idx} ({start}-{end})", fontsize=16)
    
    # 1. Histogram of r(t)
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(ratios, bins=50, range=(0.0, 1.0), alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title("Histogram of r(t) [m'(t)/m(t)]")
    ax1.set_xlabel("r(t) value (Lower is better)")
    ax1.set_ylabel("Number of Tokens")
    ax1.grid(True, alpha=0.3)
    
    # Calculate statistics
    mean_r = np.mean(ratios)
    p50_r = np.percentile(ratios, 50)
    p90_r = np.percentile(ratios, 90)
    
    stats_text = f"Mean r(t): {mean_r:.3f}\nMedian r(t): {p50_r:.3f}\n90th %ile: {p90_r:.3f}\nUnchanged (=1.0): {np.mean(ratios >= 0.99)*100:.1f}%"
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, va='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. CDF of step reduction
    ax2 = plt.subplot(2, 2, 2)
    reductions = (1.0 - ratios) * 100
    sorted_reductions = np.sort(reductions)
    p = np.arange(len(sorted_reductions)) / (len(sorted_reductions) - 1)
    
    ax2.plot(sorted_reductions, p, color='red', lw=2)
    ax2.set_title("CDF of Step Reduction (%)")
    ax2.set_xlabel("Step Reduction % (Higher is better)")
    ax2.set_ylabel("Cumulative Fraction of Tokens")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-5, 60)
    
    # Find specific points on CDF
    p_10pct_reduction = np.mean(reductions >= 10) * 100
    p_30pct_reduction = np.mean(reductions >= 30) * 100
    
    cdf_text = f"Tokens with >= 10% reduction: {p_10pct_reduction:.1f}%\nTokens with >= 30% reduction: {p_30pct_reduction:.1f}%"
    ax2.text(0.95, 0.05, cdf_text, transform=ax2.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. High-step token analysis
    ax3 = plt.subplot(2, 2, 3)
    
    # Sort tokens by m(t)
    sorted_indices = np.argsort(m_origs)[::-1] # descending
    n = len(sorted_indices)
    
    top_1pct_idx = sorted_indices[:max(1, n // 100)]
    top_10pct_idx = sorted_indices[:max(1, n // 10)]
    
    mean_r_all = np.mean(ratios)
    mean_r_top10 = np.mean(ratios[top_10pct_idx])
    mean_r_top1 = np.mean(ratios[top_1pct_idx])
    
    bars = ax3.bar(['All Tokens', 'Top 10% m(t)', 'Top 1% m(t)'], 
                   [mean_r_all, mean_r_top10, mean_r_top1], 
                   color=['gray', 'orange', 'red'])
    
    ax3.set_title("Mean r(t) by Token Magnitude")
    ax3.set_ylabel("Mean r(t) (Lower is better)")
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        reduction = (1.0 - height) * 100
        ax3.text(bar.get_x() + bar.get_width()/2., height - 0.05,
                f"r={height:.3f}\n(-{reduction:.1f}%)",
                ha='center', va='top', color='white', fontweight='bold')
                
    # 4. Scatter: m_orig vs r(t)
    ax4 = plt.subplot(2, 2, 4)
    
    # Subsample for scatter if too many points
    if len(m_origs) > 10000:
        idx = np.random.choice(len(m_origs), 10000, replace=False)
        x_scatter = m_origs[idx]
        y_scatter = ratios[idx]
    else:
        x_scatter = m_origs
        y_scatter = ratios
        
    ax4.scatter(x_scatter, y_scatter, alpha=0.1, s=5, color='purple')
    ax4.set_title("Original m(t) vs Step Reduction r(t)")
    ax4.set_xlabel("Original m(t)")
    ax4.set_ylabel("r(t) [m'(t)/m(t)]")
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    
    # Add trend line
    from scipy.stats import linregress
    try:
        slope, intercept, _, _, _ = linregress(x_scatter, y_scatter)
        x_trend = np.array([np.min(x_scatter), np.max(x_scatter)])
        y_trend = slope * x_trend + intercept
        ax4.plot(x_trend, y_trend, color='red', linestyle='--', lw=2, label='Trend')
        ax4.legend()
    except:
        pass

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    output_file = f"{output_prefix}_L{layer_idx}_H{head_idx}_B{block_idx}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_file}")
    
    return mean_r_all, mean_r_top10, mean_r_top1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('layer_dir_prefix', help='Prefix for layer dumps')
    parser.add_argument('scales_file', help='PRS scales binary file')
    parser.add_argument('--layer', type=int, default=29)
    parser.add_argument('--head', type=int, default=6)
    parser.add_argument('--output-prefix', default='step_reduction')
    
    args = parser.parse_args()
    
    print("Loading scales...")
    scales_map, n_layers, n_heads = load_prs_scales(args.scales_file)
    
    prefix_path = Path(args.layer_dir_prefix)
    layer_dir = prefix_path.parent
    file_prefix = prefix_path.name
    
    layer_file = layer_dir / f"{file_prefix}_layer_{args.layer}.bin"
    if not layer_file.exists():
        print(f"File not found: {layer_file}")
        return
        
    print(f"Loading data from {layer_file}...")
    layer_data, n_tokens = read_layer_dump(layer_file)
    if layer_data is None:
        print("Failed to read data")
        return
        
    head_data = layer_data[:, args.head, :]
    
    blocks = [(0, 32), (32, 64), (64, 96), (96, 128)]
    
    summary = []
    
    for block_idx, (start, end) in enumerate(blocks):
        print(f"\nProcessing Block {block_idx} ({start}-{end})...")
        ratios, m_origs = calculate_ratios_for_block(head_data, start, end, scales_map, args.layer, args.head)
        
        r_all, r_top10, r_top1 = plot_analysis(
            ratios, m_origs, args.layer, args.head, block_idx, start, end, args.output_prefix
        )
        
        summary.append({
            'block': block_idx,
            'start': start,
            'end': end,
            'r_all': r_all,
            'r_top10': r_top10,
            'r_top1': r_top1
        })
        
    print("\n" + "="*80)
    print("SUMMARY: MEAN r(t) ACROSS TOKENS")
    print("="*80)
    print(f"{'Block':<15} {'All Tokens':<15} {'Top 10% High-Step':<20} {'Top 1% High-Step':<20}")
    print("-" * 80)
    
    for s in summary:
        name = f"B{s['block']} ({s['start']}-{s['end']})"
        print(f"{name:<15} {s['r_all']:<15.4f} {s['r_top10']:<20.4f} {s['r_top1']:<20.4f}")

if __name__ == '__main__':
    main()
