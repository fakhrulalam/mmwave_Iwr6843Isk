"""
Configuration file for radar-based activity recognition
"""

# Data paths
DATA_DIR = "Social Behaviour"
BEHAVIORS = ["Approaching", "Sitting", "Splitting", "Standing", "Walking_togather"]

# Feature extraction parameters
TEMPORAL_WINDOW_SIZE = 36  # Use all frames (can adjust if needed)
TEMPORAL_OVERLAP = 0.5     # 50% overlap for generating multiple samples per recording

# Point cloud feature parameters
MIN_POINTS_PER_FRAME = 1   # Minimum points to consider frame valid
MAX_RANGE = 10.0           # Maximum reasonable range in meters
MIN_RANGE = 0.05           # Minimum range to avoid clutter

# Range-Doppler parameters
RD_SAMPLE_SIZE = 1000      # Sample size for range-doppler features (to manage memory)

# Feature selection
N_TOP_FEATURES = 50        # Number of top features to select
PCA_VARIANCE = 0.95        # Variance to preserve in PCA

# Model parameters
RANDOM_STATE = 42
N_FOLDS = 5                # For cross-validation
TEST_SIZE = 0.2            # If using train-test split

# Output
RESULTS_DIR = "results"
MODELS_DIR = "models"
