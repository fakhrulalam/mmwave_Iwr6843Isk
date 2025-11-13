import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

plt.rcParams.update({'font.size': 18})
plt.rcParams["figure.figsize"] = (13, 10)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def scale(doppz, Max=None, Min=None):
    """Scale doppler heatmaps to [0, 1] range"""
    if Max is None:
        Max = doppz.max()
    if Min is None:
        Min = doppz.min()
    doppz_scaled = (doppz - Min) / (Max - Min + 1e-8)
    return doppz_scaled


def load_dataset(pkl_path='processed_datasets/my_data_full.pkl'):
    """Load dataset from pickle file"""
    print(f"Loading dataset from {pkl_path}")
    df = pd.read_pickle(pkl_path)

    # Extract heatmaps and labels
    doppz = np.array(df['doppz'].values.tolist())  # Shape: (N, 64, 128)
    labels = df['activity'].values

    print(f"Original shape: {doppz.shape}")
    print(f"Labels: {np.unique(labels)}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Scale to [0, 1]
    dop_max, dop_min = doppz.max(), doppz.min()
    doppz_scaled = scale(doppz, dop_max, dop_min)

    # Add channel dimension for CNN
    doppz_scaled = doppz_scaled[..., np.newaxis]  # Shape: (N, 64, 128, 1)

    print(f"Scaled shape: {doppz_scaled.shape}")

    return doppz_scaled, labels


def get_dataset(dataset_type='full'):
    """
    Load and prepare dataset for training

    Args:
        dataset_type: 'full' (5 classes), 'dynamic' (3 classes), or 'static' (2 classes)
    """

    if dataset_type == 'full':
        pkl_file = 'processed_datasets/my_data_full.pkl'
        num_classes = 5
        label_map = {
            0: 'Approach',
            1: 'Walking',
            2: 'Splitting',
            3: 'Standing',
            4: 'Sitting'
        }
    elif dataset_type == 'dynamic':
        pkl_file = 'processed_datasets/my_data_dynamic.pkl'
        num_classes = 3
        label_map = {
            0: 'Approach',
            1: 'Walking',
            2: 'Splitting'
        }
    elif dataset_type == 'static':
        pkl_file = 'processed_datasets/my_data_static.pkl'
        num_classes = 2
        label_map = {
            3: 'Standing',
            4: 'Sitting'
        }
        # For static, remap labels to 0, 1
        X_norm, y_raw = load_dataset(pkl_file)
        y = y_raw - 3  # Remap 3->0, 4->1
        y = to_categorical(y, num_classes=num_classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y, test_size=0.3, random_state=42
        )
        label_map = {0: 'Standing', 1: 'Sitting'}
        return X_train, X_test, y_train, y_test, label_map
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Load data
    X_norm, y_raw = load_dataset(pkl_file)

    # One-hot encode labels
    y = to_categorical(y_raw, num_classes=num_classes)

    # Train-test split (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.3, random_state=42
    )

    print(f"\nDataset: {dataset_type}")
    print(f"Classes: {num_classes}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test, label_map


def get_model(num_classes=5):
    """
    Create CNN model for 64×128 heatmap input

    Args:
        num_classes: Number of output classes
    """
    model = tf.keras.Sequential([
        # Input: 64×128×1
        tf.keras.layers.Conv2D(32, (3, 5), (1, 2), padding="same",
                              activation='relu', input_shape=(64, 128, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), (2, 2), padding="same",
                              activation='relu'),
        tf.keras.layers.Conv2D(96, (3, 3), (2, 2), padding="same",
                              activation='relu'),
        tf.keras.layers.Conv2D(128, (3, 3), (2, 2), padding="same",
                              activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, "relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, "softmax")
    ])
    return model


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training history plot: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Normalize
    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_norm = np.round(conf_matrix_norm, 2)

    # Plot
    plt.figure(figsize=(10, 8))
    df_cm = pd.DataFrame(conf_matrix_norm,
                         index=[labels[i] for i in range(len(labels))],
                         columns=[labels[i] for i in range(len(labels))])
    sns.heatmap(df_cm, vmin=0, vmax=1, annot=True, cmap="Blues", fmt='.2f')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion matrix: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("Training Activity Recognition Model")
    print("="*60)

    # Choose dataset type: 'full', 'dynamic', or 'static'
    DATASET_TYPE = 'full'  # Change this to 'dynamic' or 'static' if needed
    EPOCHS = 100
    BATCH_SIZE = 16  # Smaller batch size for small dataset

    # Load dataset
    X_train, X_test, y_train, y_test, label_map = get_dataset(DATASET_TYPE)
    num_classes = len(label_map)

    # Create model
    print(f"\nCreating model for {num_classes} classes...")
    model = get_model(num_classes=num_classes)
    print(model.summary())

    # Compile model
    model.compile(
        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=["accuracy"]
    )

    # Callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"my_model_{DATASET_TYPE}_{timestamp}"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'models/{model_name}_best.weights.h5',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f'logs/{model_name}'
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    # Create directories
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Train model
    print(f"\n{'='*60}")
    print(f"Training model...")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        validation_split=0.2,  # 20% of training data for validation
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, tensorboard, early_stop],
        verbose=1
    )

    # Plot training history
    plot_training_history(history,
                         save_path=f'results/{model_name}_history.png')

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}\n")

    # Load best weights
    model.load_weights(f'models/{model_name}_best.weights.h5')

    # Predict
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Metrics
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Classification report
    print(f"\n{'='*60}")
    print("Classification Report")
    print(f"{'='*60}\n")
    target_names = [label_map[i] for i in sorted(label_map.keys())]
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, target_names,
                         save_path=f'results/{model_name}_confusion_matrix.png')

    # Save final model
    model.save(f'models/{model_name}_final.h5')
    print(f"\n✅ Model saved: models/{model_name}_final.h5")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Results saved in 'results/' folder")
    print(f"Model saved in 'models/' folder")
    print(f"TensorBoard logs in 'logs/' folder")
    print(f"\nTo view TensorBoard:")
    print(f"  tensorboard --logdir=logs/{model_name}")
