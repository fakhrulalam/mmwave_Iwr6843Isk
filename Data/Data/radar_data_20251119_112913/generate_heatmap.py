#!/usr/bin/env python3
"""
Range-Doppler Heatmap Generator
Generated automatically for data session: 20251119_112913
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def generate_gui_style_heatmap(csv_file, frame_number=1):
    """
    Generate heatmap with exact same design as the GUI.
    """
    print(f"Loading data from: {csv_file}")
    
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Get unique values
    range_bins = sorted(df['range_bin'].unique())
    doppler_bins = sorted(df['doppler_bin'].unique())
    
    print(f"Range bins: {len(range_bins)} ({min(range_bins)} to {max(range_bins)})")
    print(f"Doppler bins: {len(doppler_bins)} ({min(doppler_bins)} to {max(doppler_bins)})")
    
    # Filter data for specific frame
    frame_data = df[df['frame_number'] == frame_number]
    
    if frame_data.empty:
        print(f"No data found for frame {frame_number}")
        return
    
    # Create 2D array with proper dimensions
    max_range_bin = max(range_bins) if range_bins else 255
    max_doppler_bin = max(doppler_bins) if doppler_bins else 127
    
    heatmap = np.zeros((max_range_bin + 1, max_doppler_bin + 1))
    
    for _, row in frame_data.iterrows():
        range_bin = int(row['range_bin'])
        doppler_bin = int(row['doppler_bin'])
        if range_bin <= max_range_bin and doppler_bin <= max_doppler_bin:
            heatmap[range_bin, doppler_bin] = row['signal_strength']
    
    # Apply FFT shift to center Doppler bins around 0 (like in GUI)
    heatmap_shifted = np.fft.fftshift(heatmap, axes=1)
    
    # Create plot with GUI-style design
    plt.figure(figsize=(12, 8))
    
    # Calculate axis ranges (same as GUI)
    range_values = np.arange(0, max_range_bin + 1)
    doppler_values = np.arange(-128, 129)  # -128 to +128 like GUI
    
    # Create heatmap with GUI-style appearance
    im = plt.imshow(heatmap_shifted, cmap='jet', aspect='auto', origin='lower',
                    extent=[doppler_values[0], doppler_values[-1], 
                           range_values[0], range_values[-1]])
    
    # Add colorbar (same style as GUI)
    cbar = plt.colorbar(im)
    cbar.set_label('Signal Strength', rotation=270, labelpad=20)
    
    # Set labels and title (exact same as GUI)
    plt.xlabel('Speed [km/h]')
    plt.ylabel('Range [m]')
    plt.title('Range-Doppler Heatmap')
    
    # Add grid (same as GUI)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to match GUI
    plt.xlim(-128, 128)
    plt.ylim(0, max_range_bin)
    
    # Save plot
    output_file = f"range_doppler_heatmap_frame_{frame_number}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    # Print statistics
    print(f"\nStatistics for Frame {frame_number}:")
    print(f"  Signal strength range: {heatmap.min():.2f} to {heatmap.max():.2f}")
    print(f"  Non-zero points: {np.count_nonzero(heatmap)}")
    print(f"  Total points: {heatmap.size}")
    print(f"  Range bins: 0 to {max_range_bin}")
    print(f"  Doppler bins: -128 to +128")

def main():
    """Main function."""
    # Look for range-Doppler heatmap CSV in current directory
    csv_pattern = "range_doppler_heatmap_*.csv"
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return
    
    # Use the most recent CSV file
    csv_file = max(csv_files, key=os.path.getctime)
    print(f"Using CSV file: {csv_file}")
    
    # Generate heatmap for frame 1
    generate_gui_style_heatmap(csv_file, frame_number=1)

if __name__ == "__main__":
    main()
