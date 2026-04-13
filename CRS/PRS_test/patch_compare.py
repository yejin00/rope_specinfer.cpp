import re

with open('/home/yjkim00/rope_specinfer.cpp/scripts/compare_absmax.py', 'r') as f:
    content = f.read()

# Replace block visualization for ax1
block1_ax1 = """    # Block 0 (0-32)
    ax1.axvspan(0, 32, alpha=0.1, color='red', label='Block 0 (0-32)')
    # Block 1 (32-64)
    ax1.axvspan(32, 64, alpha=0.1, color='orange', label='Block 1 (32-64)')
    # Block 2 (64-96)
    ax1.axvspan(64, 96, alpha=0.1, color='green', label='Block 2 (64-96)')
    # Block 3 (96-128)
    ax1.axvspan(96, 128, alpha=0.1, color='blue', label='Block 3 (96-128)')"""
content = re.sub(r'    # Block 2 \(64-96\)\n    ax1\.axvspan\(64, 96, alpha=0\.1,\n                color=\'green\', label=\'Block 2 \(64-96\)\'\)\n    # Block 3 \(96-128\)\n    ax1\.axvspan\(96, 128, alpha=0\.1,\n                color=\'blue\', label=\'Block 3 \(96-128\)\'\)', block1_ax1, content)

# Replace block visualization for ax2
block1_ax2 = """    # Block 0 (0-32)
    ax2.axvspan(0, 32, alpha=0.1, color='red', label='Block 0 (0-32)')
    # Block 1 (32-64)
    ax2.axvspan(32, 64, alpha=0.1, color='orange', label='Block 1 (32-64)')
    # Block 2 (64-96)
    ax2.axvspan(64, 96, alpha=0.1, color='green', label='Block 2 (64-96)')
    # Block 3 (96-128)
    ax2.axvspan(96, 128, alpha=0.1, color='blue', label='Block 3 (96-128)')"""
content = re.sub(r'    # Block 2 \(64-96\)\n    ax2\.axvspan\(64, 96, alpha=0\.1,\n                color=\'green\', label=\'Block 2 \(64-96\)\'\)\n    # Block 3 \(96-128\)\n    ax2\.axvspan\(96, 128, alpha=0\.1,\n                color=\'blue\', label=\'Block 3 \(96-128\)\'\)', block1_ax2, content)

with open('/home/yjkim00/rope_specinfer.cpp/scripts/compare_absmax.py', 'w') as f:
    f.write(content)
