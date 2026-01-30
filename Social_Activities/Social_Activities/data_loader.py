"""
Data loading utilities for radar activity recognition
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import config


class RadarDataLoader:
    """Load and preprocess radar data from CSV files"""

    def __init__(self, data_dir: str = config.DATA_DIR):
        self.data_dir = data_dir
        self.behaviors = config.BEHAVIORS

    def load_all_data(self) -> Tuple[List[Dict], List[str], List[str]]:
        """
        Load all radar recordings from all behavior categories

        Returns:
            samples: List of dictionaries containing loaded data
            labels: List of behavior labels
            sample_names: List of sample identifiers
        """
        samples = []
        labels = []
        sample_names = []

        for behavior in self.behaviors:
            behavior_path = Path(self.data_dir) / behavior

            if not behavior_path.exists():
                print(f"Warning: {behavior_path} does not exist")
                continue

            # Find all radar_data folders
            radar_dirs = sorted([d for d in behavior_path.iterdir()
                               if d.is_dir() and d.name.startswith('radar_data_')])

            print(f"Loading {behavior}: {len(radar_dirs)} recordings")

            for radar_dir in radar_dirs:
                try:
                    sample_data = self.load_single_sample(radar_dir)
                    samples.append(sample_data)
                    labels.append(behavior)
                    sample_names.append(f"{behavior}_{radar_dir.name}")
                except Exception as e:
                    print(f"Error loading {radar_dir}: {e}")
                    continue

        print(f"\nTotal samples loaded: {len(samples)}")
        print(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

        return samples, labels, sample_names

    def load_single_sample(self, sample_dir: Path) -> Dict:
        """
        Load a single radar recording (point cloud, range-doppler, SNR)

        Args:
            sample_dir: Path to radar_data_YYYYMMDD_HHMMSS directory

        Returns:
            Dictionary with loaded data
        """
        # Load point cloud
        pc_path = sample_dir / "points_cloud.csv"
        points_cloud = pd.read_csv(pc_path)

        # Load SNR data
        snr_path = sample_dir / "noise_snr.csv"
        noise_snr = pd.read_csv(snr_path)

        # Load range-doppler (sample it to reduce memory usage)
        rd_path = sample_dir / "range_doppler.csv"
        range_doppler = pd.read_csv(rd_path)

        # Sample range-doppler if too large
        if len(range_doppler) > config.RD_SAMPLE_SIZE:
            range_doppler = range_doppler.sample(n=config.RD_SAMPLE_SIZE,
                                                 random_state=config.RANDOM_STATE)

        return {
            'points_cloud': points_cloud,
            'noise_snr': noise_snr,
            'range_doppler': range_doppler,
            'sample_dir': str(sample_dir)
        }

    def preprocess_point_cloud(self, points_cloud: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess point cloud data:
        - Remove outliers
        - Filter by range
        - Remove static clutter (zero doppler)

        Args:
            points_cloud: Raw point cloud DataFrame

        Returns:
            Cleaned point cloud DataFrame
        """
        df = points_cloud.copy()

        # Filter by range
        df = df[(df['range'] >= config.MIN_RANGE) & (df['range'] <= config.MAX_RANGE)]

        # Remove points with NaN values
        df = df.dropna()

        # Optional: Remove static clutter (very low doppler)
        # Commented out as it might remove valid sitting/standing data
        # df = df[np.abs(df['doppler']) > 0.01]

        return df

    def get_frames(self, points_cloud: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Group point cloud data by frame number

        Args:
            points_cloud: Point cloud DataFrame

        Returns:
            Dictionary mapping frame_number to DataFrame of points in that frame
        """
        frames = {}
        for frame_num in points_cloud['frame_number'].unique():
            frames[frame_num] = points_cloud[points_cloud['frame_number'] == frame_num]

        return frames


def create_temporal_windows(n_frames: int,
                            window_size: int,
                            overlap: float = 0.5) -> List[Tuple[int, int]]:
    """
    Create sliding temporal windows over frames

    Args:
        n_frames: Total number of frames
        window_size: Size of each window
        overlap: Overlap ratio (0-1)

    Returns:
        List of (start_frame, end_frame) tuples
    """
    if window_size > n_frames:
        # If window is larger than data, return single window
        return [(0, n_frames)]

    stride = max(1, int(window_size * (1 - overlap)))
    windows = []

    start = 0
    while start + window_size <= n_frames:
        windows.append((start, start + window_size))
        start += stride

    # Add final window if there's remaining data
    if start < n_frames and len(windows) > 0:
        windows.append((n_frames - window_size, n_frames))
    elif len(windows) == 0:
        windows.append((0, n_frames))

    return windows


if __name__ == "__main__":
    # Test data loading
    loader = RadarDataLoader()
    samples, labels, names = loader.load_all_data()

    print(f"\nSample data structure:")
    print(f"Point cloud shape: {samples[0]['points_cloud'].shape}")
    print(f"Point cloud columns: {samples[0]['points_cloud'].columns.tolist()}")
    print(f"Range-doppler shape: {samples[0]['range_doppler'].shape}")
    print(f"Noise SNR shape: {samples[0]['noise_snr'].shape}")
