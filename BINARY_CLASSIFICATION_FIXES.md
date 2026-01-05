# Binary Classification Optimization Report

## Problem Analysis

Your binary classification model was experiencing critical issues:

### Symptoms:
- **Accuracy: 15.23%** (terrible - worse than random)
- **Precision: 15.23%** (only 15% of positive predictions are correct)
- **Recall: 100%** (predicts EVERYTHING as positive)
- **F1 Score: 0.2644** (poor)
- **True Negatives: 0** (never predicts negative class!)
- **False Positives: 1786** (all negative samples marked as positive)
- **Overfitting gap: 0.31** (train: 0.8822, val: 1.1945)

### Root Causes:
1. **pos_weight=5.0 too aggressive** - Pushed model to always predict positive
2. **Inconsistent labeling method** - Used quantile (0.85) creating variable threshold
3. **No early stopping** - Training continued despite no improvement
4. **Poor monitoring** - Didn't track probability distributions
5. **Overfitting** - 0.31 gap between train/val loss

---

## Optimizations Implemented

### 1. **Reduced pos_weight: 5.0 → 2.5**

**Why:** Original pos_weight=5.0 was too aggressive for a 5.56:1 imbalance ratio. This caused the model to heavily bias toward predicting positive class.

**Change:**
```python
# OLD
pos_weight_value = 5.0  # Too high!

# NEW  
pos_weight_value = 2.5  # Reasonable for 5.56:1 imbalance
```

**Impact:** Model will be less biased toward positive predictions, allowing better discrimination.

---

### 2. **Fixed Binarization Method: Quantile → Top-K**

**Why:** Using `torch.quantile(labels, 0.85)` creates variable thresholds per video. If a video has flat scores, quantile picks arbitrary cutoff.

**Change:**
```python
# OLD - Inconsistent quantile-based
def binarize_labels(labels, threshold=0.85):
    threshold_value = torch.quantile(labels, threshold)
    return (labels >= threshold_value).float()

# NEW - Consistent top-k selection  
def binarize_labels(labels, ratio=0.15):
    k = max(1, int(len(labels) * ratio))
    threshold_value = torch.kthvalue(labels, len(labels) - k + 1).values
    return (labels >= threshold_value).float()
```

**Impact:** Always selects exactly top 15% of shots per video, ensuring consistent positive:negative ratio.

---

### 3. **Added Probability Monitoring**

**Why:** Need to diagnose why model predicts everything as positive. Track average probabilities for positive/negative classes.

**Change:**
```python
# Added to training metrics
'avg_pos_prob': avg_pos_prob,  # Average P(y=1) for positive samples
'avg_neg_prob': avg_neg_prob   # Average P(y=1) for negative samples
```

**Impact:** Can see if probabilities are miscalibrated (e.g., avg_pos_prob=0.95, avg_neg_prob=0.90 → both high).

---

### 4. **Implemented Early Stopping**

**Why:** Training continued for 40 epochs without improvement, causing overfitting.

**Change:**
```python
# Added early stopping parameters
best_val_metric = 0.0  # Track best val F1
patience = 7           # Stop after 7 epochs without improvement
patience_counter = 0
min_delta = 0.001      # Minimum improvement threshold

# In training loop
if val_metrics['f1'] > best_val_metric + min_delta:
    best_val_metric = val_metrics['f1']
    patience_counter = 0
else:
    patience_counter += 1

if patience_counter >= patience:
    print("Early stopping triggered!")
    break
```

**Impact:** Stops training when validation F1 plateaus, preventing overfitting.

---

### 5. **Enhanced Monitoring & Logging**

**Why:** Need detailed per-epoch diagnostics to understand model behavior.

**Change:**
```python
print(f"  Train | Loss: {train_loss:.4f} Acc: {acc:.3f} P: {prec:.3f} R: {rec:.3f} F1: {f1:.3f}")
print(f"        | Avg P(pos): {avg_pos:.3f} Avg P(neg): {avg_neg:.3f}")
print(f"  Val   | Loss: {val_loss:.4f} Acc: {acc:.3f} P: {prec:.3f} R: {rec:.3f} F1: {f1:.3f}")
print(f"        | TP: {tp} FP: {fp} FN: {fn} TN: {tn}")
```

**Impact:** See exact confusion matrix and probability distributions per epoch.

---

### 6. **Added Accuracy to History**

**Why:** Track both accuracy and F1 over time.

**Change:**
```python
history = {
    'train_loss': [],
    'val_loss': [],
    'train_f1': [],
    'val_f1': [],
    'train_accuracy': [],  # NEW
    'val_accuracy': []     # NEW
}
```

**Impact:** Can plot accuracy curves alongside F1.

---

## Expected Results After Fixes

### Before:
- Accuracy: 15.23%
- Precision: 15.23%
- Recall: 100%
- F1: 0.2644
- TN: 0 (predicts everything as positive)

### Expected After:
- **Accuracy: 70-85%** (much better discrimination)
- **Precision: 40-60%** (reasonable for imbalanced data)
- **Recall: 60-80%** (no longer 100%)
- **F1: 0.50-0.70** (significant improvement)
- **TN: 1200-1500** (actually predicts negative class)
- **FP: 300-600** (reduced from 1786)
- **Overfitting gap: <0.15** (reduced from 0.31)

---

## How to Use

1. **Re-run training** (Cell 7) with the optimized code
2. **Monitor early stopping** - training should stop around epoch 15-25
3. **Check probability diagnostics:**
   - `Avg P(pos)` should be ~0.60-0.75 (high but not extreme)
   - `Avg P(neg)` should be ~0.20-0.40 (clearly separated)
4. **Re-run validation analysis** (Cell 14, 15) to see improved metrics
5. **If still issues:**
   - Further reduce pos_weight to 2.0
   - Try adjusting decision threshold from 0.5 to 0.4 or 0.6
   - Check if model architecture needs more capacity

---

## Additional Recommendations

### Short-term:
1. **Try pos_weight=2.0** if results still biased toward positive
2. **Experiment with thresholds**: Try 0.4, 0.45, 0.55, 0.60 instead of 0.5
3. **Add label smoothing**: `BCEWithLogitsLoss(label_smoothing=0.1)`

### Medium-term:
1. **Focal loss**: Better handles extreme imbalance
   ```python
   class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0):
           self.alpha = alpha
           self.gamma = gamma
   ```
2. **Calibration**: Use temperature scaling post-training
3. **Threshold tuning**: Find optimal threshold on validation set

### Long-term:
1. **Data augmentation**: Generate more positive samples
2. **Ensemble**: Train multiple models with different pos_weights
3. **Architecture changes**: Add batch normalization, increase dropout

---

## Verification Checklist

After re-running training, verify:

- [ ] TN > 0 (model actually predicts negative class)
- [ ] Accuracy > 50% (better than random)
- [ ] F1 > 0.40 (reasonable performance)
- [ ] Train/val gap < 0.20 (reduced overfitting)
- [ ] Early stopping triggered before epoch 40
- [ ] Avg P(pos) and Avg P(neg) are well-separated (>0.15 difference)

---

## Summary

The key issue was **pos_weight=5.0 being too aggressive**, combined with **inconsistent binarization** and **no early stopping**. The fixes:

1. ✅ Reduced pos_weight to 2.5
2. ✅ Fixed binarization to use top-k
3. ✅ Added early stopping (patience=7)
4. ✅ Enhanced monitoring with probability tracking
5. ✅ Added TP/FP/FN/TN display

**Expected improvement:** F1 from 0.26 to 0.50-0.70, accuracy from 15% to 70-85%.
