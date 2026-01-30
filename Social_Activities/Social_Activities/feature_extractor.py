"""
Feature extraction for radar-based activity recognition
Based on research findings for point cloud, range-doppler, and temporal features
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from typing import Dict, List
import config


class RadarFeatureExtractor:
    """Extract comprehensive features from radar data"""

    def __init__(self):
        self.feature_names = []

    def extract_features(self, sample_data: Dict) -> np.ndarray:
        """
        Extract all features from a single radar sample

        Args:
            sample_data: Dictionary containing points_cloud, noise_snr, range_doppler

        Returns:
            Feature vector as numpy array
        """
        features = {}

        # Extract point cloud features
        pc_features = self.extract_point_cloud_features(sample_data['points_cloud'])
        features.update(pc_features)

        # Extract range-doppler features
        rd_features = self.extract_range_doppler_features(sample_data['range_doppler'])
        features.update(rd_features)

        # Extract SNR features
        snr_features = self.extract_snr_features(sample_data['noise_snr'])
        features.update(snr_features)

        # Extract temporal features
        temporal_features = self.extract_temporal_features(sample_data['points_cloud'])
        features.update(temporal_features)

        # Store feature names (only once)
        if not self.feature_names:
            self.feature_names = list(features.keys())

        return np.array([features[name] for name in self.feature_names])

    def extract_point_cloud_features(self, points_cloud: pd.DataFrame) -> Dict:
        """
        Extract spatial and velocity features from point cloud

        Features include:
        - Spatial statistics (x, y, z)
        - Velocity statistics (doppler)
        - Range statistics
        - Angle of arrival statistics
        - Point count features
        """
        features = {}

        if len(points_cloud) == 0:
            return self._empty_point_cloud_features()

        # === SPATIAL FEATURES ===
        for coord in ['x', 'y', 'z']:
            features[f'{coord}_mean'] = points_cloud[coord].mean()
            features[f'{coord}_std'] = points_cloud[coord].std()
            features[f'{coord}_min'] = points_cloud[coord].min()
            features[f'{coord}_max'] = points_cloud[coord].max()
            features[f'{coord}_range'] = points_cloud[coord].max() - points_cloud[coord].min()
            features[f'{coord}_median'] = points_cloud[coord].median()
            features[f'{coord}_skew'] = stats.skew(points_cloud[coord])
            features[f'{coord}_kurtosis'] = stats.kurtosis(points_cloud[coord])

        # Spatial spread (2D)
        features['spatial_spread_xy'] = np.sqrt(points_cloud['x'].std()**2 + points_cloud['y'].std()**2)
        features['spatial_spread_xyz'] = np.sqrt(points_cloud['x'].std()**2 +
                                                  points_cloud['y'].std()**2 +
                                                  points_cloud['z'].std()**2)

        # Centroid
        features['centroid_x'] = points_cloud['x'].mean()
        features['centroid_y'] = points_cloud['y'].mean()
        features['centroid_distance'] = np.sqrt(features['centroid_x']**2 + features['centroid_y']**2)

        # Bounding box volume
        features['bbox_volume'] = (features['x_range'] *
                                  features['y_range'] *
                                  max(features['z_range'], 0.01))  # Avoid zero

        # Point density
        features['point_density'] = len(points_cloud) / max(features['bbox_volume'], 0.01)

        # === VELOCITY FEATURES (Doppler) ===
        features['doppler_mean'] = points_cloud['doppler'].mean()
        features['doppler_std'] = points_cloud['doppler'].std()
        features['doppler_min'] = points_cloud['doppler'].min()
        features['doppler_max'] = points_cloud['doppler'].max()
        features['doppler_range'] = points_cloud['doppler'].max() - points_cloud['doppler'].min()
        features['doppler_median'] = points_cloud['doppler'].median()
        features['doppler_abs_mean'] = np.abs(points_cloud['doppler']).mean()
        features['doppler_abs_max'] = np.abs(points_cloud['doppler']).max()
        features['doppler_skew'] = stats.skew(points_cloud['doppler'])
        features['doppler_kurtosis'] = stats.kurtosis(points_cloud['doppler'])

        # Doppler energy
        features['doppler_energy'] = np.sum(points_cloud['doppler']**2)
        features['doppler_rms'] = np.sqrt(np.mean(points_cloud['doppler']**2))

        # Zero-crossing rate (static vs dynamic indicator)
        doppler_vals = points_cloud['doppler'].values
        features['doppler_zero_crossings'] = np.sum(np.diff(np.sign(doppler_vals)) != 0) / len(doppler_vals)

        # === RANGE FEATURES ===
        features['range_mean'] = points_cloud['range'].mean()
        features['range_std'] = points_cloud['range'].std()
        features['range_min'] = points_cloud['range'].min()
        features['range_max'] = points_cloud['range'].max()
        features['range_range'] = points_cloud['range'].max() - points_cloud['range'].min()
        features['range_median'] = points_cloud['range'].median()
        features['range_skew'] = stats.skew(points_cloud['range'])
        features['range_kurtosis'] = stats.kurtosis(points_cloud['range'])

        # === ANGLE OF ARRIVAL FEATURES ===
        features['aoa_mean'] = points_cloud['aoa'].mean()
        features['aoa_std'] = points_cloud['aoa'].std()
        features['aoa_min'] = points_cloud['aoa'].min()
        features['aoa_max'] = points_cloud['aoa'].max()
        features['aoa_range'] = points_cloud['aoa'].max() - points_cloud['aoa'].min()
        features['aoa_median'] = points_cloud['aoa'].median()

        # Angular spread (important for multi-person detection)
        features['angular_spread'] = points_cloud['aoa'].std()

        # === POINT COUNT FEATURES ===
        features['total_points'] = len(points_cloud)
        features['points_per_frame'] = len(points_cloud) / max(points_cloud['frame_number'].nunique(), 1)

        # Points per frame statistics
        points_per_frame = points_cloud.groupby('frame_number').size()
        features['points_per_frame_std'] = points_per_frame.std()
        features['points_per_frame_min'] = points_per_frame.min()
        features['points_per_frame_max'] = points_per_frame.max()

        return features

    def extract_range_doppler_features(self, range_doppler: pd.DataFrame) -> Dict:
        """
        Extract features from range-doppler maps

        Features include:
        - Energy statistics
        - Distribution features
        - Spectral features
        """
        features = {}

        if len(range_doppler) == 0:
            return self._empty_range_doppler_features()

        signal = range_doppler['signal_strength'].values

        # === ENERGY FEATURES ===
        features['rd_total_energy'] = np.sum(signal)
        features['rd_mean_energy'] = np.mean(signal)
        features['rd_std_energy'] = np.std(signal)
        features['rd_max_energy'] = np.max(signal)
        features['rd_min_energy'] = np.min(signal)

        # Energy concentration
        sorted_signal = np.sort(signal)[::-1]
        top_10_percent = int(0.1 * len(signal))
        features['rd_energy_concentration'] = np.sum(sorted_signal[:top_10_percent]) / features['rd_total_energy']

        # Energy entropy
        signal_prob = signal / (features['rd_total_energy'] + 1e-10)
        signal_prob = signal_prob[signal_prob > 0]
        features['rd_energy_entropy'] = -np.sum(signal_prob * np.log(signal_prob + 1e-10))

        # === DISTRIBUTION FEATURES ===
        features['rd_skew'] = stats.skew(signal)
        features['rd_kurtosis'] = stats.kurtosis(signal)

        # Range bin statistics
        features['rd_range_bin_mean'] = range_doppler['range_bin'].mean()
        features['rd_range_bin_std'] = range_doppler['range_bin'].std()

        # Doppler bin statistics
        features['rd_doppler_bin_mean'] = range_doppler['doppler_bin'].mean()
        features['rd_doppler_bin_std'] = range_doppler['doppler_bin'].std()

        # === SPECTRAL FEATURES ===
        # Weighted centroid in range-doppler space
        total_strength = features['rd_total_energy']
        features['rd_range_centroid'] = np.sum(range_doppler['range_bin'] * signal) / total_strength
        features['rd_doppler_centroid'] = np.sum(range_doppler['doppler_bin'] * signal) / total_strength

        # Spread in range-doppler space
        features['rd_range_spread'] = np.sqrt(np.sum(
            (range_doppler['range_bin'] - features['rd_range_centroid'])**2 * signal) / total_strength)
        features['rd_doppler_spread'] = np.sqrt(np.sum(
            (range_doppler['doppler_bin'] - features['rd_doppler_centroid'])**2 * signal) / total_strength)

        # Peak location
        max_idx = signal.argmax()
        features['rd_peak_range_bin'] = range_doppler.iloc[max_idx]['range_bin']
        features['rd_peak_doppler_bin'] = range_doppler.iloc[max_idx]['doppler_bin']

        return features

    def extract_snr_features(self, noise_snr: pd.DataFrame) -> Dict:
        """
        Extract signal quality features from SNR data
        """
        features = {}

        if len(noise_snr) == 0:
            return self._empty_snr_features()

        # SNR statistics
        features['snr_mean'] = noise_snr['snr'].mean()
        features['snr_std'] = noise_snr['snr'].std()
        features['snr_min'] = noise_snr['snr'].min()
        features['snr_max'] = noise_snr['snr'].max()
        features['snr_median'] = noise_snr['snr'].median()

        # Noise statistics
        features['noise_mean'] = noise_snr['noise'].mean()
        features['noise_std'] = noise_snr['noise'].std()
        features['noise_min'] = noise_snr['noise'].min()
        features['noise_max'] = noise_snr['noise'].max()

        # Detection quality
        high_snr_threshold = 30  # dB
        features['high_snr_ratio'] = (noise_snr['snr'] > high_snr_threshold).mean()

        return features

    def extract_temporal_features(self, points_cloud: pd.DataFrame) -> Dict:
        """
        Extract temporal dynamics features

        Features include:
        - Trajectory features
        - Velocity/acceleration over time
        - Temporal statistics
        """
        features = {}

        if len(points_cloud) == 0 or points_cloud['frame_number'].nunique() < 2:
            return self._empty_temporal_features()

        # Group by frame
        frames = points_cloud.groupby('frame_number')

        # Compute per-frame statistics
        frame_stats = frames.agg({
            'x': ['mean', 'std'],
            'y': ['mean', 'std'],
            'doppler': ['mean', 'std', 'max'],
            'range': ['mean', 'std', 'min'],
            'aoa': ['mean', 'std']
        })

        # Flatten column names
        frame_stats.columns = ['_'.join(col).strip() for col in frame_stats.columns.values]
        frame_stats = frame_stats.reset_index()

        # === TRAJECTORY FEATURES ===
        # Centroid trajectory
        x_trajectory = frame_stats['x_mean'].values
        y_trajectory = frame_stats['y_mean'].values

        # Path length (total distance traveled by centroid)
        if len(x_trajectory) > 1:
            path_length = np.sum(np.sqrt(np.diff(x_trajectory)**2 + np.diff(y_trajectory)**2))
            features['trajectory_path_length'] = path_length

            # Net displacement
            displacement = np.sqrt((x_trajectory[-1] - x_trajectory[0])**2 +
                                 (y_trajectory[-1] - y_trajectory[0])**2)
            features['trajectory_displacement'] = displacement

            # Tortuosity (path length / displacement)
            features['trajectory_tortuosity'] = path_length / (displacement + 1e-10)

            # Velocity (centroid velocity)
            features['trajectory_velocity_mean'] = np.mean(np.sqrt(np.diff(x_trajectory)**2 + np.diff(y_trajectory)**2))
            features['trajectory_velocity_std'] = np.std(np.sqrt(np.diff(x_trajectory)**2 + np.diff(y_trajectory)**2))
        else:
            features['trajectory_path_length'] = 0
            features['trajectory_displacement'] = 0
            features['trajectory_tortuosity'] = 1
            features['trajectory_velocity_mean'] = 0
            features['trajectory_velocity_std'] = 0

        # === TEMPORAL DYNAMICS ===
        # Doppler temporal statistics
        doppler_mean_series = frame_stats['doppler_mean'].values
        features['temporal_doppler_trend'] = np.polyfit(range(len(doppler_mean_series)), doppler_mean_series, 1)[0]
        features['temporal_doppler_std'] = np.std(doppler_mean_series)
        features['temporal_doppler_change'] = doppler_mean_series[-1] - doppler_mean_series[0] if len(doppler_mean_series) > 0 else 0

        # Range temporal statistics (approaching vs receding)
        range_mean_series = frame_stats['range_mean'].values
        features['temporal_range_trend'] = np.polyfit(range(len(range_mean_series)), range_mean_series, 1)[0]
        features['temporal_range_std'] = np.std(range_mean_series)
        features['temporal_range_change'] = range_mean_series[-1] - range_mean_series[0] if len(range_mean_series) > 0 else 0

        # Angle temporal statistics (movement direction)
        aoa_mean_series = frame_stats['aoa_mean'].values
        features['temporal_aoa_trend'] = np.polyfit(range(len(aoa_mean_series)), aoa_mean_series, 1)[0]
        features['temporal_aoa_std'] = np.std(aoa_mean_series)
        features['temporal_aoa_change'] = aoa_mean_series[-1] - aoa_mean_series[0] if len(aoa_mean_series) > 0 else 0

        # Spatial spread over time
        x_std_series = frame_stats['x_std'].values
        y_std_series = frame_stats['y_std'].values
        spatial_spread_series = np.sqrt(x_std_series**2 + y_std_series**2)
        features['temporal_spatial_spread_mean'] = np.mean(spatial_spread_series)
        features['temporal_spatial_spread_std'] = np.std(spatial_spread_series)
        features['temporal_spatial_spread_trend'] = np.polyfit(range(len(spatial_spread_series)), spatial_spread_series, 1)[0]

        # Number of frames
        features['num_frames'] = points_cloud['frame_number'].nunique()

        return features

    def _empty_point_cloud_features(self) -> Dict:
        """Return zero features for empty point cloud"""
        feature_names = [
            'x_mean', 'x_std', 'x_min', 'x_max', 'x_range', 'x_median', 'x_skew', 'x_kurtosis',
            'y_mean', 'y_std', 'y_min', 'y_max', 'y_range', 'y_median', 'y_skew', 'y_kurtosis',
            'z_mean', 'z_std', 'z_min', 'z_max', 'z_range', 'z_median', 'z_skew', 'z_kurtosis',
            'spatial_spread_xy', 'spatial_spread_xyz', 'centroid_x', 'centroid_y', 'centroid_distance',
            'bbox_volume', 'point_density',
            'doppler_mean', 'doppler_std', 'doppler_min', 'doppler_max', 'doppler_range',
            'doppler_median', 'doppler_abs_mean', 'doppler_abs_max', 'doppler_skew', 'doppler_kurtosis',
            'doppler_energy', 'doppler_rms', 'doppler_zero_crossings',
            'range_mean', 'range_std', 'range_min', 'range_max', 'range_range',
            'range_median', 'range_skew', 'range_kurtosis',
            'aoa_mean', 'aoa_std', 'aoa_min', 'aoa_max', 'aoa_range', 'aoa_median', 'angular_spread',
            'total_points', 'points_per_frame', 'points_per_frame_std',
            'points_per_frame_min', 'points_per_frame_max'
        ]
        return {name: 0.0 for name in feature_names}

    def _empty_range_doppler_features(self) -> Dict:
        """Return zero features for empty range-doppler"""
        feature_names = [
            'rd_total_energy', 'rd_mean_energy', 'rd_std_energy', 'rd_max_energy', 'rd_min_energy',
            'rd_energy_concentration', 'rd_energy_entropy', 'rd_skew', 'rd_kurtosis',
            'rd_range_bin_mean', 'rd_range_bin_std', 'rd_doppler_bin_mean', 'rd_doppler_bin_std',
            'rd_range_centroid', 'rd_doppler_centroid', 'rd_range_spread', 'rd_doppler_spread',
            'rd_peak_range_bin', 'rd_peak_doppler_bin'
        ]
        return {name: 0.0 for name in feature_names}

    def _empty_snr_features(self) -> Dict:
        """Return zero features for empty SNR"""
        feature_names = [
            'snr_mean', 'snr_std', 'snr_min', 'snr_max', 'snr_median',
            'noise_mean', 'noise_std', 'noise_min', 'noise_max', 'high_snr_ratio'
        ]
        return {name: 0.0 for name in feature_names}

    def _empty_temporal_features(self) -> Dict:
        """Return zero features for insufficient temporal data"""
        feature_names = [
            'trajectory_path_length', 'trajectory_displacement', 'trajectory_tortuosity',
            'trajectory_velocity_mean', 'trajectory_velocity_std',
            'temporal_doppler_trend', 'temporal_doppler_std', 'temporal_doppler_change',
            'temporal_range_trend', 'temporal_range_std', 'temporal_range_change',
            'temporal_aoa_trend', 'temporal_aoa_std', 'temporal_aoa_change',
            'temporal_spatial_spread_mean', 'temporal_spatial_spread_std', 'temporal_spatial_spread_trend',
            'num_frames'
        ]
        return {name: 0.0 for name in feature_names}


if __name__ == "__main__":
    # Test feature extraction
    from data_loader import RadarDataLoader

    loader = RadarDataLoader()
    samples, labels, names = loader.load_all_data()

    extractor = RadarFeatureExtractor()

    # Extract features from first sample
    features = extractor.extract_features(samples[0])
    print(f"Extracted {len(features)} features")
    print(f"Feature names: {extractor.feature_names[:10]}... (showing first 10)")
    print(f"Feature values: {features[:10]}")
