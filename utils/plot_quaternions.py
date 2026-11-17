#!/usr/bin/env python3
"""
Plot a single comparison of the original and optimized quaternion paths.
Only output one figure with four subplots (w, x, y, z).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.ticker as ticker
from pathlib import Path

# One place for every font size in the figure
plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 12,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
})


def load_quaternion_data(original_path, optimized_path):
    """Load the original and optimized quaternion series."""
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
    return result

def clip_window(data, start_s, dur_s):
    """Keep only one window of the series, rebased so the window starts at t=0.

    Optimized rows are frame-aligned with the original, so one index mask serves both.
    Output timestamps stay in nanoseconds for the 1 s locator.
    """
    # Every adapter writes the time column as integer nanoseconds
    T = data['original']['T']
    t_sec = (T - T[0]) * 1e-9
    total = float(t_sec[-1])
    if not dur_s or dur_s > total:
        dur_s = total
    start = float(start_s) if start_s is not None else max(0.0, (total - dur_s) / 2.0)
    idx = np.flatnonzero((t_sec >= start) & (t_sec <= start + dur_s))
    if len(idx) < 3:
        raise SystemExit(f'window {start:.2f}+{dur_s:.2f}s holds {len(idx)} samples; '
                         f'the sequence is {total:.2f}s long')

    out = {'original': dict(data['original']), 'optimized': dict(data['optimized'])}
    out['original']['T'] = (t_sec[idx] - start) * 1e9
    for key in ('w', 'x', 'y', 'z'):
        out['original'][key] = data['original'][key][idx]
    out['original']['frames'] = len(idx)
    n_opt = data['optimized']['frames']
    for key in ('w', 'x', 'y', 'z'):
        out['optimized'][key] = data['optimized'][key][idx[idx < n_opt]]
    out['optimized']['frames'] = int(min(len(idx), n_opt))
    return out


def plot_quaternion_comparison(data, save_path, png_name='quaternion_comparison.png'):
    """
    Plot a single comparison figure with four subplots for w, x, y, z components.
    Style inspired by plot_quat_compare.
    """
    original = data['original']
    optimized = data['optimized']
    T = original['T']

    # Figure width proportional to the time span (original style),
    # capped at ~20k px at save dpi=150 so long clips stay readable.
    SAVE_DPI = 150
    target_px = (T[-1] - T[0]) / 8000000000 * 4000
    target_px = min(18000.0, target_px)
    width = target_px / SAVE_DPI
    height = 3600.0 / SAVE_DPI
    
    sub_fig_num = 4
    fig, ax = plt.subplots(sub_fig_num, 1, sharex=True, figsize=(width, height))
    plt.tight_layout()
    
    components = ['w', 'x', 'y', 'z']
    component_labels = ['W', 'X', 'Y', 'Z']
    
    # X in seconds; nanoseconds would render as 0.0-1.0 with a 1e9 offset label
    T_sec = (T - T[0]) * 1e-9
    n_opt = optimized['frames']

    for i, comp in enumerate(components):
        # Optimized is frame-aligned with the original, so it shares the same time axis
        ax[i].plot(T_sec, original[comp], '.-', alpha=0.5, label='Original')
        ax[i].plot(T_sec[:n_opt], optimized[comp], '.-', alpha=0.5, label='Optimized')
        ax[i].tick_params(axis='y', left=False, labelleft=False)
        ax[i].text(0.006, 0.05, component_labels[i], transform=ax[i].transAxes,
                   va='bottom', ha='left')
        ax[i].legend()
    
    ax[-1].set_xlabel('Time (s)')
    
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(save_path, png_name)
    plt.savefig(output_path, dpi=SAVE_DPI)
    plt.close()
    
    return output_path

def main():
    """Main function: load two quaternion CSVs and save a comparison plot."""
    ap = argparse.ArgumentParser(description='Plot original vs optimized quaternions.')
    base_dir = Path(__file__).parent.parent
    # Defaults point at the first rendered DVS clip so a bare run just works
    dvs_clip = Path('/home/tangjian5/data/DeepFVS/dataset_release/output/pipeline/'
                    's_114b_outdoor_running_trail_night')
    ap.add_argument('--original', default=str(dvs_clip / 'quat.csv'))
    ap.add_argument('--optimized', default=str(dvs_clip / 'opt.csv'))
    ap.add_argument('--out-dir', default='/tmp/plotquat')
    ap.add_argument('--png-name', default='dvs_trail_night.png')
    ap.add_argument('--clip-sec', type=float, default=0.0,
                    help='plotted window in seconds (0 = whole sequence)')
    ap.add_argument('--start-sec', type=float, default=None,
                    help='window start in seconds (default: centred)')
    args = ap.parse_args()

    data = load_quaternion_data(args.original, args.optimized)

    print(f"\nData Summary:")
    print(f"  Original frames: {data['original']['frames']}")
    print(f"  Optimized frames: {data['optimized']['frames']}")

    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison plot, windowed to --clip-sec unless 0 was passed
    data = clip_window(data, args.start_sec, args.clip_sec)
    saved_path = plot_quaternion_comparison(data, save_path=output_dir, png_name=args.png_name)
    print(f"\nSaved plot to: {saved_path}")


if __name__ == '__main__':
    main()
