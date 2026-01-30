"""
Inference script for making predictions on new radar data
"""

import numpy as np
import joblib
from pathlib import Path
import sys

from data_loader import RadarDataLoader
import config


class RadarPredictor:
    """Make predictions on new radar data using trained models"""

    def __init__(self, model_path: str):
        """
        Initialize predictor with trained model

        Args:
            model_path: Path to saved model (.pkl file)
        """
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)

        # Load preprocessing objects
        models_dir = Path(model_path).parent
        self.scaler = joblib.load(models_dir / "scaler.pkl")
        self.label_encoder = joblib.load(models_dir / "label_encoder.pkl")
        self.feature_selector = joblib.load(models_dir / "feature_selector.pkl")
        self.extractor = joblib.load(models_dir / "feature_extractor.pkl")

        print("Model loaded successfully!")

    def predict_sample(self, sample_data: dict) -> tuple:
        """
        Predict activity for a single radar sample

        Args:
            sample_data: Dictionary with 'points_cloud', 'noise_snr', 'range_doppler'

        Returns:
            (predicted_class, probabilities)
        """
        # Extract features
        features = self.extractor.extract_features(sample_data)
        features = features.reshape(1, -1)

        # Handle NaN/inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize
        features_scaled = self.scaler.transform(features)

        # Select features
        features_selected = self.feature_selector.transform(features_scaled)

        # Predict
        prediction = self.model.predict(features_selected)[0]
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]

        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_selected)[0]
            prob_dict = {class_name: prob
                        for class_name, prob in zip(self.label_encoder.classes_, probabilities)}
        else:
            prob_dict = None

        return predicted_class, prob_dict

    def predict_directory(self, radar_dir: str):
        """
        Predict activity for a radar recording directory

        Args:
            radar_dir: Path to directory containing points_cloud.csv, etc.

        Returns:
            (predicted_class, probabilities)
        """
        loader = RadarDataLoader()
        sample_data = loader.load_single_sample(Path(radar_dir))

        return self.predict_sample(sample_data)


def main():
    """Command-line interface for prediction"""
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model_path> <radar_data_directory>")
        print("\nExample:")
        print("  python predict.py models/random_forest_model.pkl 'Social Behaviour/Approaching/radar_data_20251209_171833'")
        sys.exit(1)

    model_path = sys.argv[1]
    radar_dir = sys.argv[2]

    # Create predictor
    predictor = RadarPredictor(model_path)

    # Make prediction
    print(f"\nPredicting activity for: {radar_dir}")
    predicted_class, probabilities = predictor.predict_directory(radar_dir)

    print("\n" + "=" * 60)
    print(f"PREDICTED ACTIVITY: {predicted_class}")
    print("=" * 60)

    if probabilities:
        print("\nClass probabilities:")
        for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")


if __name__ == "__main__":
    main()
