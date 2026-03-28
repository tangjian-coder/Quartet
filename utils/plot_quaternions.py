#!/usr/bin/env python3
"""
Plot single comparison of original and filtered quaternion paths.
Only output one figure with four subplots (w, x, y, z).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
from pathlib import Path

def load_quaternion_data():
    """Load original and filtered quaternion data."""
    # Paths
    base_dir = Path(__file__).parent.parent
    original_path = base_dir / "data" / "DbgFile_Frame_OriginalCamOrient.csv"
    filtered_path = base_dir / "data" / "optimized_quaternions.csv"
    
    print(f"Loading original quaternions from: {original_path}")
    print(f"Loading filtered quaternions from: {filtered_path}")
    
    # Load original data (has time column?)
    original_df = pd.read_csv(original_path)
    print(f"Original data shape: {original_df.shape}")
    print(f"Original columns: {list(original_df.columns)}")
    
    # Load filtered data (no time column, just w,x,y,z)
    filtered_df = pd.read_csv(filtered_path, header=None, names=['w', 'x', 'y', 'z'])
    print(f"Filtered data shape: {filtered_df.shape}")
    
    # Extract quaternion components
    original_w = original_df['w'].values
    original_x = original_df['x'].values
    original_y = original_df['y'].values
    original_z = original_df['z'].values
    
    filtered_w = filtered_df['w'].values
    filtered_x = filtered_df['x'].values
    filtered_y = filtered_df['y'].values
    filtered_z = filtered_df['z'].values
    
    # If original data has a timestamp column, use it; otherwise use frame index
    time_column = None
    for col in original_df.columns:
        if 'time' in col.lower() or 'timestamp' in col.lower():
            time_column = col
            break
    if time_column is not None:
        T = original_df[time_column].values
    else:
        T = np.arange(len(original_w))
    
    return {
        'original': {
            'w': original_w,
            'x': original_x,
            'y': original_y,
            'z': original_z,
            'frames': len(original_w),
            'T': T
        },
        'filtered': {
            'w': filtered_w,
            'x': filtered_x,
            'y': filtered_y,
            'z': filtered_z,
            'frames': len(filtered_w)
        }
    }

def plot_quaternion_comparison(data, save_path, png_name='quaternion_comparison.png', fps=30):
    """
    Plot a single comparison figure with four subplots for w, x, y, z components.
    Style inspired by plot_quat_compare.
    """
    original = data['original']
    filtered = data['filtered']
    T = original['T']
    
    # Calculate figure width similar to reference function
    import matplotlib as mpl
    dpi = mpl.rcParams['figure.dpi']
    width = (T[-1] - T[0]) * 4000 / (dpi * (16000000000 if fps == 30 else 8000000000))
    width = min(30000 / dpi, width)
    height = 6000 / dpi
    
    sub_fig_num = 4
    fig, ax = plt.subplots(sub_fig_num, 1, sharex=True, figsize=(width, height))
    for i in range(sub_fig_num):
        ax[i].grid(which='both')
    plt.tight_layout()
    
    components = ['w', 'x', 'y', 'z']
    component_labels = ['W', 'X', 'Y', 'Z']
    
    for i, comp in enumerate(components):
        # Original data (all frames)
        ax[i].plot(T - T[0], original[comp], '.-', alpha=0.5, label='Original')
        # Filtered data (may have fewer frames)
        filtered_frames = np.arange(filtered['frames'])
        # Align time: assume filtered corresponds to first N original timestamps
        if filtered['frames'] <= len(T):
            T_filtered = T[:filtered['frames']]
        else:
            T_filtered = np.arange(filtered['frames']) * (T[1] - T[0]) + T[0]
        ax[i].plot(T_filtered - T[0], filtered[comp], '.-', alpha=0.5, label='Filtered')
        ax[i].set_ylabel(f'{component_labels[i]}', fontsize=12)
        ax[i].legend()
    
    ax[-1].set_xlabel('Time (relative)', fontsize=12)
    
    # Set x-axis major locator similar to reference
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1000000000))
    
    plt.suptitle('Quaternion Components: Original vs Filtered', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save figure
    output_path = os.path.join(save_path, png_name)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def main():
    """Main function to load data and create a single comparison plot."""
    print("Quaternion Comparison Script")
    print("=" * 50)
    
    # Load data
    data = load_quaternion_data()
    
    print(f"\nData Summary:")
    print(f"  Original frames: {data['original']['frames']}")
    print(f"  Filtered frames: {data['filtered']['frames']}")
    
    # Create output directory for plots
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Generate single comparison plot
    print("\nGenerating single comparison plot...")
    saved_path = plot_quaternion_comparison(
        data, 
        save_path=output_dir,
        png_name='quaternion_comparison.png',
        fps=30
    )
    
    print(f"  Saved plot to: {saved_path}")
    print(f"\nPlot saved to: {output_dir}")
    
    # Optionally show the plot if display is available
    if 'DISPLAY' in os.environ:
        plt.show()
    else:
        print("\nNo display available. Plot saved to disk.")

if __name__ == "__main__":
    main()