"""
Main training script for radar-based activity recognition
Includes feature selection, model training, and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, f1_score)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
import joblib
import os
from pathlib import Path
from tqdm import tqdm

import config
from data_loader import RadarDataLoader
from feature_extractor import RadarFeatureExtractor

# Try to import XGBoost, fallback if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class RadarActivityRecognition:
    """Complete pipeline for radar-based activity recognition"""

    def __init__(self):
        self.loader = RadarDataLoader()
        self.extractor = RadarFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.pca = None

        # Create output directories
        Path(config.RESULTS_DIR).mkdir(exist_ok=True)
        Path(config.MODELS_DIR).mkdir(exist_ok=True)

    def load_and_extract_features(self):
        """Load data and extract features from all samples"""
        print("=" * 60)
        print("LOADING DATA AND EXTRACTING FEATURES")
        print("=" * 60)

        # Load all samples
        samples, labels, sample_names = self.loader.load_all_data()

        # Extract features
        print("\nExtracting features...")
        X = []
        for i, sample in enumerate(tqdm(samples, desc="Feature extraction")):
            features = self.extractor.extract_features(sample)
            X.append(features)

        X = np.array(X)
        y = np.array(labels)

        print(f"\nFeature extraction complete!")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {len(self.extractor.feature_names)}")
        print(f"Number of samples: {len(X)}")

        # Check for NaN or inf values
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"Warning: Found {nan_count} NaN and {inf_count} inf values. Replacing with 0.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y, sample_names

    def preprocess_features(self, X_train, X_test=None):
        """Normalize features using StandardScaler"""
        print("\nNormalizing features...")

        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def select_features(self, X, y, method='mutual_info', k=None):
        """
        Feature selection using mutual information or PCA

        Args:
            X: Feature matrix
            y: Labels
            method: 'mutual_info' or 'pca'
            k: Number of features to select (default from config)
        """
        if k is None:
            k = min(config.N_TOP_FEATURES, X.shape[1])

        print(f"\nFeature selection ({method})...")
        print(f"Selecting top {k} features from {X.shape[1]}")

        if method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)

            # Get selected feature names and scores
            selected_indices = self.feature_selector.get_support(indices=True)
            scores = self.feature_selector.scores_[selected_indices]
            selected_features = [self.extractor.feature_names[i] for i in selected_indices]

            # Sort by importance
            feature_importance = sorted(zip(selected_features, scores),
                                       key=lambda x: x[1], reverse=True)

            print("\nTop 20 most important features:")
            for feat, score in feature_importance[:20]:
                print(f"  {feat}: {score:.4f}")

            # Save feature importance
            importance_df = pd.DataFrame(feature_importance,
                                        columns=['Feature', 'Importance'])
            importance_df.to_csv(f"{config.RESULTS_DIR}/feature_importance.csv", index=False)

        elif method == 'pca':
            self.pca = PCA(n_components=k)
            X_selected = self.pca.fit_transform(X)

            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")

        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"Selected feature shape: {X_selected.shape}")
        return X_selected

    def train_models(self, X, y):
        """Train multiple models with cross-validation"""
        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=config.RANDOM_STATE
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=config.RANDOM_STATE
            ),
            'SVM (Linear)': SVC(
                kernel='linear',
                C=1.0,
                random_state=config.RANDOM_STATE
            ),
            'SVM (Poly)': SVC(
                kernel='poly',
                degree=3,
                C=1.0,
                gamma='scale',
                random_state=config.RANDOM_STATE
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean',
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=config.RANDOM_STATE
            )
        }

        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                eval_metric='mlogloss'
            )

        # Cross-validation
        cv = StratifiedKFold(n_splits=config.N_FOLDS,
                            shuffle=True,
                            random_state=config.RANDOM_STATE)

        results = {}

        for name, model in models.items():
            print(f"\n{name}:")
            print("-" * 40)

            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y_encoded, cv=cv,
                                       scoring='accuracy', n_jobs=-1)

            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            # Get predictions for confusion matrix
            y_pred = cross_val_predict(model, X, y_encoded, cv=cv, n_jobs=-1)

            # Calculate per-class metrics
            accuracy = accuracy_score(y_encoded, y_pred)
            f1 = f1_score(y_encoded, y_pred, average='weighted')

            print(f"Overall accuracy: {accuracy:.4f}")
            print(f"Weighted F1-score: {f1:.4f}")

            # Classification report
            class_names = self.label_encoder.classes_
            report = classification_report(y_encoded, y_pred,
                                          target_names=class_names,
                                          output_dict=True)

            print("\nPer-class metrics:")
            for class_name in class_names:
                metrics = report[class_name]
                print(f"  {class_name}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    F1-score: {metrics['f1-score']:.4f}")

            # Store results
            results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'accuracy': accuracy,
                'f1_score': f1,
                'y_pred': y_pred,
                'report': report
            }

            # Train final model on all data
            print(f"Training final {name} model on all data...")
            model.fit(X, y_encoded)

            # Save model
            model_path = f"{config.MODELS_DIR}/{name.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")

        return results, y_encoded

    def plot_confusion_matrices(self, results, y_true):
        """Plot confusion matrices for all models"""
        print("\nGenerating confusion matrices...")

        class_names = self.label_encoder.classes_
        n_models = len(results)

        # Arrange in grid (max 4 columns)
        n_cols = min(4, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # Flatten axes array for easier indexing
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes

        for idx, (name, result) in enumerate(results.items()):
            cm = confusion_matrix(y_true, result['y_pred'])

            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx], cbar=True, vmin=0, vmax=1)
            axes[idx].set_title(f'{name}\nAcc: {result["accuracy"]:.3f}', fontsize=10)
            axes[idx].set_ylabel('True Label', fontsize=9)
            axes[idx].set_xlabel('Predicted Label', fontsize=9)
            axes[idx].tick_params(labelsize=8)

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_DIR}/confusion_matrices.png", dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {config.RESULTS_DIR}/confusion_matrices.png")
        plt.close()

    def plot_performance_comparison(self, results):
        """Plot performance comparison across models"""
        print("Generating performance comparison plot...")

        # Prepare data and sort by accuracy
        model_data = []
        for name in results.keys():
            model_data.append({
                'name': name,
                'accuracy': results[name]['accuracy'],
                'f1_score': results[name]['f1_score'],
                'cv_mean': results[name]['cv_scores'].mean(),
                'cv_std': results[name]['cv_scores'].std()
            })

        # Sort by CV accuracy (most reliable metric)
        model_data.sort(key=lambda x: x['cv_mean'], reverse=True)

        model_names = [m['name'] for m in model_data]
        accuracies = [m['accuracy'] for m in model_data]
        f1_scores = [m['f1_score'] for m in model_data]
        cv_means = [m['cv_mean'] for m in model_data]
        cv_stds = [m['cv_std'] for m in model_data]

        # Create horizontal bar chart for better readability
        fig, ax = plt.subplots(figsize=(12, max(8, len(model_names) * 0.6)))

        y_pos = np.arange(len(model_names))
        height = 0.25

        # Plot horizontal bars
        ax.barh(y_pos - height, accuracies, height, label='Test Accuracy', alpha=0.8)
        ax.barh(y_pos, f1_scores, height, label='F1-Score', alpha=0.8)
        ax.barh(y_pos + height, cv_means, height, label='CV Accuracy', alpha=0.8,
               xerr=cv_stds, capsize=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.invert_yaxis()  # Best models at top
        ax.set_xlabel('Score')
        ax.set_title('Model Performance Comparison (Sorted by CV Accuracy)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (acc, f1, cv) in enumerate(zip(accuracies, f1_scores, cv_means)):
            ax.text(cv + 0.02, i + height, f'{cv:.3f}', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{config.RESULTS_DIR}/performance_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to {config.RESULTS_DIR}/performance_comparison.png")
        plt.close()

    def save_results(self, results):
        """Save detailed results to CSV"""
        print("Saving results...")

        # Model comparison
        model_comparison = []
        for name, result in results.items():
            model_comparison.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'F1-Score': result['f1_score'],
                'CV Mean': result['cv_scores'].mean(),
                'CV Std': result['cv_scores'].std()
            })

        df_comparison = pd.DataFrame(model_comparison)
        df_comparison.to_csv(f"{config.RESULTS_DIR}/model_comparison.csv", index=False)
        print(f"Model comparison saved to {config.RESULTS_DIR}/model_comparison.csv")

        # Detailed per-class results
        for name, result in results.items():
            report_df = pd.DataFrame(result['report']).transpose()
            report_df.to_csv(f"{config.RESULTS_DIR}/{name.replace(' ', '_').lower()}_report.csv")

        # Save metadata
        metadata = {
            'total_samples': len(results[list(results.keys())[0]]['y_pred']),
            'n_features_original': len(self.extractor.feature_names),
            'n_features_selected': config.N_TOP_FEATURES,
            'n_folds': config.N_FOLDS,
            'behaviors': config.BEHAVIORS
        }

        with open(f"{config.RESULTS_DIR}/metadata.txt", 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

    def run(self):
        """Run complete pipeline"""
        print("\n" + "=" * 60)
        print("RADAR ACTIVITY RECOGNITION PIPELINE")
        print("=" * 60)

        # Step 1: Load data and extract features
        X, y, sample_names = self.load_and_extract_features()

        # Step 2: Preprocess (normalize)
        X_normalized = self.preprocess_features(X)

        # Step 3: Feature selection
        X_selected = self.select_features(X_normalized, y, method='mutual_info')

        # Step 4: Train models
        results, y_encoded = self.train_models(X_selected, y)

        # Step 5: Visualizations
        self.plot_confusion_matrices(results, y_encoded)
        self.plot_performance_comparison(results)

        # Step 6: Save results
        self.save_results(results)

        # Save preprocessing objects
        joblib.dump(self.scaler, f"{config.MODELS_DIR}/scaler.pkl")
        joblib.dump(self.label_encoder, f"{config.MODELS_DIR}/label_encoder.pkl")
        joblib.dump(self.feature_selector, f"{config.MODELS_DIR}/feature_selector.pkl")
        joblib.dump(self.extractor, f"{config.MODELS_DIR}/feature_extractor.pkl")

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\nResults saved to: {config.RESULTS_DIR}/")
        print(f"Models saved to: {config.MODELS_DIR}/")

        return results


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = RadarActivityRecognition()
    results = pipeline.run()
