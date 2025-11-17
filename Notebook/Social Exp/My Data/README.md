# My Data - Group Activity Recognition Dataset

## ✅ Data Processing Complete

All your CSV data has been converted and processed into training-ready format!

## Directory Structure

```
My Data/
├── Dataset 2/Group Activity/    # Original CSV data (5 activities)
├── raw_datasets/                # Converted JSON .txt files
├── processed_datasets/          # Training-ready .pkl files
│   ├── my_data_full.pkl        # All 5 activities (402 frames)
│   ├── my_data_dynamic.pkl     # Dynamic activities (244 frames)
│   └── my_data_static.pkl      # Static activities (158 frames)
└── process_data.py              # Data processing script
```

## Dataset Summary

| Metric | Value |
|--------|-------|
| **Total Frames** | 402 |
| **Activities** | 5 (Approach, Walking, Splitting, Standing, Sitting) |
| **Heatmap Shape** | 64 × 128 |
| **Data Source** | mmWave Radar |

## Activity Distribution

### Dynamic Activities (244 frames)
- **Walking**: 83 frames (label=1)
- **Splitting**: 81 frames (label=2)
- **Approach**: 80 frames (label=0)

### Static Activities (158 frames)
- **Sitting**: 81 frames (label=4)
- **Standing**: 77 frames (label=3)

## Dataset Files

### 1. my_data_full.pkl (25 MB)
- Contains all 5 activities
- 402 total frames
- Use this for 5-class classification

### 2. my_data_dynamic.pkl (15 MB)
- Dynamic/moving activities
- 244 frames (approach, walking, splitting)
- Use for 3-class dynamic activity recognition

### 3. my_data_static.pkl (10 MB)
- Static/stationary activities
- 158 frames (standing, sitting)
- Use for 2-class static activity recognition

## Data Format

### DataFrame Structure
Each .pkl file contains a pandas DataFrame with columns:
- `datetime`: Timestamp
- `rangeIdx`: Range bin index
- `dopplerIdx`: Doppler bin index
- `numDetectedObj`: Number of detected objects/points
- `range`: Distance measurement (meters)
- `peakVal`: Peak signal strength
- `x_coord`: X coordinates of detected points (list)
- `y_coord`: Y coordinates of detected points (list)
- **`doppz`**: 64×128 Range-Doppler heatmap (numpy array)
- `activity`: Activity label (0-4)
- `activity_type`: 'dynamic' or 'static'

### Range-Doppler Heatmap
- **Shape**: (64, 128) - [range_bins × doppler_bins]
- **Type**: Float32
- **Values**: Signal strength at each range-velocity combination

## How to Load Your Data

```python
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

# Load full dataset
df = pd.read_pickle('processed_datasets/my_data_full.pkl')

# Extract features and labels
X = np.array(df['doppz'].tolist())     # Shape: (402, 64, 128)
y = df['activity'].values               # Shape: (402,)

# Reshape for CNN (add channel dimension)
X = X[..., np.newaxis]                  # Shape: (402, 64, 128, 1)

# One-hot encode labels
y = to_categorical(y, num_classes=5)   # Shape: (402, 5)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

## Activity Label Mapping

```python
activity_map = {
    0: 'Approach',
    1: 'Walking',
    2: 'Splitting',
    3: 'Standing',
    4: 'Sitting'
}
```

## Next Steps

### 1. Create a CNN Classifier

Your heatmap shape is **64×128**, which is different from the original mmDoppler models. You need to create a custom classifier:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    # Input: 64×128 heatmap
    tf.keras.layers.Conv2D(32, (3, 5), (1, 2), padding="same",
                          activation='relu', input_shape=(64, 128, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), (2, 2), padding="same",
                          activation='relu'),
    tf.keras.layers.Conv2D(96, (3, 3), (2, 2), padding="same",
                          activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, "relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, "softmax")  # 5 classes
])

model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=["accuracy"]
)
```

### 2. Train Your Model

```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    batch_size=32,
    verbose=1
)
```

### 3. Evaluate

```python
from sklearn.metrics import classification_report, confusion_matrix

# Predict
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrics
print(classification_report(y_true, y_pred,
                          target_names=list(activity_map.values())))
print(confusion_matrix(y_true, y_pred))
```

## File Sizes & Performance

- **Raw JSON**: ~25 MB total (5 files)
- **Processed PKL**: ~50 MB total (3 files)
- **Heatmap Memory**: Each heatmap is ~32 KB (64×128×4 bytes)
- **Full Dataset Memory**: ~13 MB (402 heatmaps)

## Notes

- All activities are currently classified as either 'dynamic' or 'static'
- You can modify `map_activity_type()` in `process_data.py` to change classifications
- The dataset is relatively small (402 frames) - consider data augmentation
- Heatmap shape (64×128) is unique to your data collection setup

---

**Created**: 2025-11-04
**Status**: ✅ Ready for Training
**Framework**: TensorFlow/Keras compatible
