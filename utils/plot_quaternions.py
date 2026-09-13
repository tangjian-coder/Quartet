#!/usr/bin/env python3
"""
Plot single comparison of original and filtered quaternion paths.
Only output one figure with four subplots (w, x, y, z).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.ticker as ticker
from pathlib import Path


def load_quaternion_data(original_path, optimized_path, filtered_path=None):
    """Load original, optimized and (optionally) product-filtered quaternions."""
    print(f"Loading original quaternions from: {original_path}")
    print(f"Loading optimized quaternions from: {optimized_path}")

    # Load original data (has time column?)
    original_df = pd.read_csv(original_path)
    print(f"Original data shape: {original_df.shape}")
    print(f"Original columns: {list(original_df.columns)}")

    # Load optimized data (no time column, just w,x,y,z)
    optimized_df = pd.read_csv(optimized_path, header=None, names=['w', 'x', 'y', 'z'])
    print(f"Optimized data shape: {optimized_df.shape}")

    # Extract quaternion components
    original_w = original_df['w'].values
    original_x = original_df['x'].values
    original_y = original_df['y'].values
    original_z = original_df['z'].values

    optimized_w = optimized_df['w'].values
    optimized_x = optimized_df['x'].values
    optimized_y = optimized_df['y'].values
    optimized_z = optimized_df['z'].values

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

    result = {
        'original': {
            'w': original_w,
            'x': original_x,
            'y': original_y,
            'z': original_z,
            'frames': len(original_w),
            'T': T
        },
        'optimized': {
            'w': optimized_w,
            'x': optimized_x,
            'y': optimized_y,
            'z': optimized_z,
            'frames': len(optimized_w)
        }
    }
    if filtered_path is not None:
        print(f"Loading filtered quaternions from: {filtered_path}")
        filtered_df = pd.read_csv(filtered_path)
        result['filtered'] = {
            'w': filtered_df['w'].values,
            'x': filtered_df['x'].values,
            'y': filtered_df['y'].values,
            'z': filtered_df['z'].values,
            'frames': len(filtered_df),
        }
    return result

def plot_quaternion_comparison(data, save_path, png_name='quaternion_comparison.png', fps=30):
    """
    Plot a single comparison figure with four subplots for w, x, y, z components.
    Style inspired by plot_quat_compare.
    """
    original = data['original']
    optimized = data['optimized']
    T = original['T']

    # Figure width proportional to the time span (original style),
    # capped at ~20k px at save dpi=150 so long clips stay readable.
    import matplotlib as mpl
    dpi = mpl.rcParams['figure.dpi']
    width = (T[-1] - T[0]) * 4000 / (dpi * (1600000000 if fps == 30 else 8000000000))
    width = min(30000 / dpi, width)
    width = min(width, 18000.0 / 150.0)
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
        # Product-filtered data (if provided; different coordinate frame!)
        filtered = data.get('filtered')
        if filtered is not None:
            if filtered['frames'] <= len(T):
                T_filtered = T[:filtered['frames']]
            else:
                T_filtered = np.arange(filtered['frames']) * (T[1] - T[0]) + T[0]
            ax[i].plot(T_filtered - T[0], filtered[comp], '.-', alpha=0.5, label='Filtered(\u4ea7\u54c1)')
        # Optimized data (may have fewer frames)
        optimized_frames = np.arange(optimized['frames'])
        # Align time: assume optimized corresponds to first N original timestamps
        if optimized['frames'] <= len(T):
            T_optimized = T[:optimized['frames']]
        else:
            T_optimized = np.arange(optimized['frames']) * (T[1] - T[0]) + T[0]
        ax[i].plot(T_optimized - T[0], optimized[comp], '.-', alpha=0.5, label='Optimized')
        ax[i].set_ylabel(f'{component_labels[i]}', fontsize=12)
        ax[i].legend()
    
    ax[-1].set_xlabel('Time (relative)', fontsize=12)
    
    # Set x-axis major locator similar to reference
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1000000000))
    
    title = ('Quaternion Components: Original vs Filtered vs Optimized'
             if data.get('filtered') is not None else 'Quaternion Components: Original vs Optimized')
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save figure
    output_path = os.path.join(save_path, png_name)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path

def main():
    """Main function: load two quaternion CSVs and save a comparison plot."""
    ap = argparse.ArgumentParser(description='Plot original vs optimized quaternions.')
    base_dir = Path(__file__).parent.parent
    ap.add_argument('--original', default=str(base_dir / 'data' / 'DbgFile_Frame_OriginalCamOrient.csv'))
    ap.add_argument('--filtered', default=str(base_dir / 'data' / 'DbgFile_Frame_FilteredCamOrient.csv'))
    ap.add_argument('--optimized', default=str(base_dir / 'data' / 'optimized_quaternions.csv'))
    ap.add_argument('--out-dir', default=str(Path(__file__).parent / 'plots'))
    ap.add_argument('--png-name', default='quaternion_comparison.png')
    args = ap.parse_args()

    # Load data (filtered is optional: GoPro runs only have raw + optimized)
    filtered_path = args.filtered if args.filtered and os.path.exists(args.filtered) else None
    data = load_quaternion_data(args.original, args.optimized, filtered_path)

    print(f"\nData Summary:")
    print(f"  Original frames: {data['original']['frames']}")
    if 'filtered' in data:
        print(f"  Filtered frames: {data['filtered']['frames']}")
    print(f"  Optimized frames: {data['optimized']['frames']}")

    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison plot
    saved_path = plot_quaternion_comparison(data, save_path=output_dir, png_name=args.png_name)
    print(f"\nSaved plot to: {saved_path}")


if __name__ == '__main__':
    main()
