# How to Increase Accuracy WITHOUT Data Augmentation

## Current Problem

Your model has **14.05% accuracy** because:
- **Too many parameters**: 193,701 parameters with only 281 training samples
- **Parameter-to-sample ratio**: 689:1 (should be < 20:1)
- **Model is too complex** for the small dataset

## Solutions Implemented in `train_classifier_improved.py`

### 1. **Simplified Model Architecture** ⭐ MOST IMPORTANT

**Problem**: Original model has 193,701 parameters for 281 samples

**Solution**: Created two simpler models:

#### Simple Model (~5,000-10,000 parameters)
```python
Conv2D(16)  # Instead of 32
Conv2D(32)  # Instead of 64
Conv2D(48)  # Instead of 96
Dense(32)   # Instead of 64
```
- **Parameters**: ~7,000
- **Ratio**: ~25 params per sample ✅
- **Expected improvement**: 25-40% accuracy

#### Very Simple Model (~2,000-3,000 parameters)
```python
Conv2D(12)  # Minimal filters
Conv2D(24)
Dense(16)   # Very small dense layer
```
- **Parameters**: ~2,500
- **Ratio**: ~9 params per sample ✅
- **Expected improvement**: 30-50% accuracy (better for very small datasets)

**Comparison**:
| Model | Parameters | Ratio | Expected Accuracy |
|-------|-----------|-------|-------------------|
| Original | 193,701 | 689:1 | 14% (actual) |
| Simple | ~7,000 | 25:1 | 25-40% |
| Very Simple | ~2,500 | 9:1 | 30-50% |

---

### 2. **Batch Normalization**

**Added**: BatchNormalization layers after each Conv2D

**Benefits**:
- Stabilizes training
- Allows higher learning rates
- Reduces internal covariate shift
- Acts as mild regularization

---

### 3. **L2 Regularization**

**Added**: `kernel_regularizer=tf.keras.regularizers.l2(0.001)`

**Benefits**:
- Prevents overfitting
- Encourages smaller weights
- Works better than dropout alone for small datasets

---

### 4. **Stronger Dropout**

**Changed**: Dropout rates from 0.2-0.3 → 0.3-0.5

**Reasoning**:
- Small datasets need stronger regularization
- Prevents memorization
- Forces model to learn general patterns

---

### 5. **Class Weights**

**Added**: Automatic class weight computation

```python
class_weights = compute_class_weight('balanced', ...)
```

**Benefits**:
- Handles slight class imbalance
- Prevents model from always predicting majority class
- Your current model only predicts "Walking" - this fixes it

**Your class distribution**:
- Walking: 83 samples (21%) ← Model stuck here
- Sitting: 81 samples (20%)
- Splitting: 81 samples (20%)
- Approach: 80 samples (20%)
- Standing: 77 samples (19%)

---

### 6. **Learning Rate Scheduler**

**Added**: ReduceLROnPlateau callback

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10
)
```

**Benefits**:
- Reduces learning rate when training plateaus
- Helps escape local minima
- Enables finer optimization in later epochs

---

### 7. **Stratified Train-Test Split**

**Added**: `stratify=y_raw` in train_test_split

**Benefits**:
- Ensures each split has proportional class distribution
- Prevents test set from having skewed classes
- More reliable validation metrics

---

### 8. **Increased Early Stopping Patience**

**Changed**: Patience from 20 → 30 epochs

**Reasoning**:
- Small datasets need more epochs to converge
- Learning can be noisy with few samples
- Prevents premature stopping

---

### 9. **Smaller Batch Size**

**Changed**: Default batch size from 16 → 8

**Benefits**:
- More parameter updates per epoch
- Noisier gradients = better generalization
- Works better for small datasets

---

### 10. **More Aggressive Downsampling**

**Strategy**: Reduce spatial dimensions faster

**Original**:
```python
Conv2D(32, (3, 5), (1, 2))  # Slow reduction
Conv2D(64, (3, 3), (2, 2))
Conv2D(96, (3, 3), (2, 2))
Conv2D(128, (3, 3), (2, 2))
```

**Improved**:
```python
Conv2D(16, (3, 5), (2, 4))  # Fast reduction
Conv2D(32, (3, 3), (2, 2))
Conv2D(48, (3, 3), (2, 2))
```

**Benefits**:
- Fewer parameters in early layers
- Forces model to learn essential features
- Reduces overfitting

---

## How to Use the Improved Script

### Quick Start

```bash
cd "/Users/amirus/Documents/Doppler/My Data"
source /Users/amirus/Documents/Doppler/mmDoppler/venv/bin/activate
python3 train_classifier_improved.py
```

### Configuration Options

Edit these variables in the script:

```python
DATASET_TYPE = 'full'      # 'full', 'dynamic', 'static'
MODEL_SIZE = 'simple'      # 'simple', 'very_simple'
EPOCHS = 200
BATCH_SIZE = 8             # Try 4, 8, 16
LEARNING_RATE = 0.001      # Try 0.0005, 0.001, 0.002
```

---

## Recommended Experiments

### Experiment 1: Very Simple Model, Full Dataset ⭐ TRY THIS FIRST
```python
DATASET_TYPE = 'full'
MODEL_SIZE = 'very_simple'
BATCH_SIZE = 8
LEARNING_RATE = 0.001
```
**Expected**: 30-45% accuracy

---

### Experiment 2: Simple Model, Full Dataset
```python
DATASET_TYPE = 'full'
MODEL_SIZE = 'simple'
BATCH_SIZE = 8
LEARNING_RATE = 0.001
```
**Expected**: 25-40% accuracy

---

### Experiment 3: Binary Classification (Easier Task) ⭐ BEST FOR SMALL DATA
```python
DATASET_TYPE = 'static'     # Only Standing vs Sitting
MODEL_SIZE = 'very_simple'
BATCH_SIZE = 4
LEARNING_RATE = 0.001
```
**Expected**: 50-70% accuracy (much easier with 2 classes!)

---

### Experiment 4: 3-Class Problem
```python
DATASET_TYPE = 'dynamic'    # Approach, Walking, Splitting
MODEL_SIZE = 'very_simple'
BATCH_SIZE = 8
LEARNING_RATE = 0.001
```
**Expected**: 40-55% accuracy

---

### Experiment 5: Lower Learning Rate
```python
DATASET_TYPE = 'full'
MODEL_SIZE = 'very_simple'
BATCH_SIZE = 8
LEARNING_RATE = 0.0005      # Slower learning
```
**Expected**: Slower but potentially more stable training

---

### Experiment 6: Smaller Batches
```python
DATASET_TYPE = 'full'
MODEL_SIZE = 'very_simple'
BATCH_SIZE = 4              # Even smaller
LEARNING_RATE = 0.001
```
**Expected**: Noisier gradients, better generalization

---

## Expected Results

### Realistic Accuracy Targets (WITHOUT Data Augmentation)

| Dataset | Classes | Expected Accuracy | Baseline (Random) |
|---------|---------|-------------------|-------------------|
| Full | 5 | **30-50%** | 20% |
| Dynamic | 3 | **40-60%** | 33% |
| Static | 2 | **60-80%** | 50% |

**Note**: These are realistic for 402 samples without augmentation.

---

## Why These Improvements Work

### The Core Problem: Parameter-to-Data Ratio

Deep learning needs a good ratio:

| Scenario | Parameters | Samples | Ratio | Result |
|----------|-----------|---------|-------|--------|
| **Ideal** | 10,000 | 10,000 | 1:1 | Great! |
| **Good** | 5,000 | 1,000 | 5:1 | OK |
| **Acceptable** | 5,000 | 250 | 20:1 | Marginal |
| **YOUR ORIGINAL** | 193,701 | 281 | **689:1** | Failed ❌ |
| **IMPROVED (Simple)** | 7,000 | 281 | **25:1** | Marginal ✅ |
| **IMPROVED (Very Simple)** | 2,500 | 281 | **9:1** | Good ✅ |

### Why Original Model Failed

1. **Too many parameters** → Model memorizes training data
2. **Can't generalize** → Poor test performance
3. **Collapses to majority class** → Only predicts "Walking"
4. **Overfitting** → 100% train accuracy, 14% test accuracy

### How Improvements Fix This

1. **Fewer parameters** → Less capacity to overfit
2. **Regularization (L2 + Dropout)** → Prevents memorization
3. **Class weights** → Prevents majority class bias
4. **Batch normalization** → Stabilizes training
5. **Learning rate scheduling** → Better convergence

---

## If Accuracy Is Still Low

### Option A: Simplify the Task

Instead of 5 classes, try:
- **Binary**: Static (Standing/Sitting) vs Dynamic (others) → 70-80% expected
- **3-way**: Group similar activities → 50-65% expected

### Option B: Traditional Machine Learning

With small datasets, classical ML often beats deep learning:

1. Extract features from heatmaps:
   - Mean, std, max, min
   - Histogram statistics
   - Texture features (GLCM)
   - Frequency domain features (FFT)

2. Train Random Forest or SVM:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   # Extract features
   # Train RF → Expected 50-65% accuracy
   ```

**Advantage**: Classical ML needs much less data!

### Option C: Collect More Data ⭐ BEST SOLUTION

- **Current**: 402 samples
- **Target**: 2,000-5,000 samples (400-1000 per class)
- **Impact**: Could reach 70-90% accuracy

---

## Monitoring Training

### View TensorBoard

```bash
tensorboard --logdir=logs/improved_very_simple_full_XXXXXXXX_XXXXXX
```

Open: http://localhost:6006

### What to Look For

**Good Training**:
- Training accuracy: 40-60%
- Validation accuracy: 30-50%
- Gap (train - val): < 15%

**Overfitting** (bad):
- Training accuracy: 90%
- Validation accuracy: 20%
- Gap: 70% ← Model memorizing!

**Underfitting** (bad):
- Training accuracy: 20%
- Validation accuracy: 18%
- Model too simple or needs more epochs

---

## Summary of Changes

| Aspect | Original | Improved | Impact |
|--------|----------|----------|--------|
| **Parameters** | 193,701 | 2,500-7,000 | ⭐⭐⭐ HUGE |
| **Param/Sample Ratio** | 689:1 | 9-25:1 | ⭐⭐⭐ HUGE |
| **Regularization** | Dropout only | L2 + Dropout + BN | ⭐⭐ Large |
| **Class Weights** | No | Yes | ⭐⭐ Large |
| **LR Scheduling** | No | Yes | ⭐ Medium |
| **Stratified Split** | No | Yes | ⭐ Small |
| **Batch Size** | 16 | 8 | ⭐ Small |
| **Early Stop Patience** | 20 | 30 | ⭐ Small |

---

## Next Steps

### Step 1: Run Experiment 1 (Binary Classification)
```bash
python3 train_classifier_improved.py
# Edit: DATASET_TYPE = 'static', MODEL_SIZE = 'very_simple'
```
**Expected**: 60-75% accuracy

### Step 2: Run Experiment 2 (Full Dataset)
```bash
python3 train_classifier_improved.py
# Edit: DATASET_TYPE = 'full', MODEL_SIZE = 'very_simple'
```
**Expected**: 30-50% accuracy

### Step 3: Compare Results
Look at:
- Test accuracy
- Confusion matrix (is it predicting all classes now?)
- Training curves (any overfitting?)

---

## Questions?

Common issues:

**Q: Still only predicting one class?**
A: Try smaller model (`very_simple`) or fewer classes (`static`)

**Q: Training accuracy very low (< 30%)?**
A: Increase LEARNING_RATE to 0.002 or EPOCHS to 300

**Q: Large gap between train and validation accuracy?**
A: Increase dropout rates or L2 regularization

**Q: Model not improving after epoch 1?**
A: Use smaller batch size (4) or lower learning rate (0.0005)

---

## Comparison with mmDoppler Paper

| Metric | mmDoppler | Your Data (Before) | Your Data (Target) |
|--------|-----------|-------------------|-------------------|
| Dataset Size | 75,000 | 402 | 402 (no change) |
| Parameters | 193,701 | 193,701 | 2,500-7,000 ✅ |
| Accuracy | 95% | 14% | 30-50% ✅ |
| Classes Predicted | All 19 | Only 1/5 | All 5 ✅ |

**Key Insight**: mmDoppler had 186x more data. You need a 77x simpler model (193,701 → 2,500) to compensate!

---

Good luck! Start with **Experiment 1** or **Experiment 3** for best results.
