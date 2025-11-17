# Quick Start Guide - Improved Training with Automatic Report

## What's New?

The improved training script now **automatically generates a comprehensive results report** after each training run!

## How to Use

### 1. Run the Training

```bash
cd "/Users/amirus/Documents/Doppler/My Data"
source /Users/amirus/Documents/Doppler/mmDoppler/venv/bin/activate
python3 train_classifier_improved.py
```

### 2. What Gets Generated Automatically

After training completes, you'll get:

```
📊 Generated Files:
├── models/
│   ├── improved_[size]_[type]_[timestamp]_best.weights.h5   # Best model weights
│   └── improved_[size]_[type]_[timestamp]_final.h5          # Final model
├── results/
│   ├── improved_[size]_[type]_[timestamp]_history.png       # Training curves
│   ├── improved_[size]_[type]_[timestamp]_confusion_matrix.png  # Confusion matrix
│   └── improved_[size]_[type]_[timestamp]_REPORT.md         # ⭐ COMPREHENSIVE REPORT
└── logs/
    └── improved_[size]_[type]_[timestamp]/                  # TensorBoard logs
```

### 3. Read Your Report

The report will be at: `results/improved_[size]_[type]_[timestamp]_REPORT.md`

It contains:
- ✅ Executive Summary (accuracy, improvement, parameters)
- ✅ Training Configuration
- ✅ Performance Metrics
- ✅ Before/After Comparison (14.05% → NEW%)
- ✅ Per-Class Performance Analysis
- ✅ Confusion Patterns
- ✅ What Worked / What Could Be Better
- ✅ Personalized Recommendations
- ✅ Experiment Suggestions

## Configuration Options

Edit these variables in `train_classifier_improved.py`:

```python
# Line ~740
DATASET_TYPE = 'full'      # Options: 'full', 'dynamic', 'static'
MODEL_SIZE = 'simple'      # Options: 'simple', 'very_simple'
EPOCHS = 200
BATCH_SIZE = 8             # Try: 4, 8, 16
LEARNING_RATE = 0.001      # Try: 0.0005, 0.001, 0.002
```

## Recommended First Run

For best results with your small dataset, try:

```python
DATASET_TYPE = 'full'
MODEL_SIZE = 'very_simple'  # Smallest model, best for 402 samples
BATCH_SIZE = 8
LEARNING_RATE = 0.001
```

**Expected**: 30-50% accuracy (vs 14.05% baseline)

## Alternative: Binary Classification

For highest accuracy, try binary classification:

```python
DATASET_TYPE = 'static'     # Only Standing vs Sitting
MODEL_SIZE = 'very_simple'
BATCH_SIZE = 4
LEARNING_RATE = 0.001
```

**Expected**: 60-75% accuracy

## Report Highlights

The auto-generated report will tell you:

### Performance
- **Test Accuracy**: X.XX%
- **Improvement**: +XX% over baseline
- **Classes Predicted**: X/5 (vs 1/5 before)

### Analysis
- Training Status (overfitting/underfitting/good)
- Which classes are confused
- Parameter efficiency
- Generalization quality

### Recommendations
Based on your actual results, the report provides:
- Specific next steps
- Hyperparameter suggestions
- Alternative approaches
- Experiment ideas

## Example Output

After running, you'll see:

```
============================================================
Training Complete!
============================================================
Model: very_simple
Dataset: full (5 classes)
Total Parameters: 2,549
Params/Sample Ratio: 9.1
Test Accuracy: 35.54%

📊 Files Generated:
  - Model: models/improved_very_simple_full_20251104_143022_final.h5
  - Weights: models/improved_very_simple_full_20251104_143022_best.weights.h5
  - Report: results/improved_very_simple_full_20251104_143022_REPORT.md ⭐
  - Plots: results/improved_very_simple_full_20251104_143022_history.png
  - Confusion: results/improved_very_simple_full_20251104_143022_confusion_matrix.png
  - Logs: logs/improved_very_simple_full_20251104_143022/

📖 To view the detailed report:
  Open: results/improved_very_simple_full_20251104_143022_REPORT.md

📈 To view TensorBoard:
  tensorboard --logdir=logs/improved_very_simple_full_20251104_143022
  Then open: http://localhost:6006
============================================================
```

## Key Improvements Over Original

| Feature | Original Script | Improved Script |
|---------|----------------|-----------------|
| **Parameters** | 193,701 | 2,500-7,000 |
| **Ratio** | 689:1 | 9-25:1 ✅ |
| **Regularization** | Dropout only | L2 + Dropout + BN |
| **Class Handling** | No | Balanced weights |
| **Report** | Manual | **Auto-generated** ⭐ |

## Troubleshooting

### If accuracy is still < 25%
The report will recommend:
- Switch to binary classification (`DATASET_TYPE = 'static'`)
- Try traditional ML
- Adjust hyperparameters

### If overfitting (train >> val accuracy)
The report will suggest:
- Use 'very_simple' model
- Increase dropout
- Stronger L2 regularization

### If not predicting all classes
The report will recommend:
- Check class weights
- Reduce to fewer classes
- Collect more data

## Questions?

All analysis and recommendations are **automatically included in the report** based on your actual training results!

---

**Happy Training!** 🚀
