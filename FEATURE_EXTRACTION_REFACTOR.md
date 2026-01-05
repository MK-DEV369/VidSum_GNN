# Feature Extraction Refactoring - Complete

## Summary

**Goal**: Shift from computing heuristic importance scores to extracting rich, descriptive features that a GNN can learn from.

---

## Step 1: Removed Importance Computation ✅

Disabled the following functions:
- ❌ `compute_importance()` 
- ❌ `emphasize_peaks()`
- ❌ `smooth_importance_scores()`
- ❌ `assign_ranks()` (importance-dependent)

Replaced with deprecated stubs that raise clear errors:
- `compute_importance_DEPRECATED()`
- `emphasize_peaks_DEPRECATED()`
- `smooth_importance_scores_DEPRECATED()`

These functions will tell developers to use the GNN instead of baked-in heuristics.

---

## Step 2: Redesigned ShotFeatures ✅

### Old (Limited):
```python
class ShotFeatures:
    motion: float
    speech: float
    scene_change: float
    audio_energy: float
    object_count: float
```

### New (Rich descriptors):
```python
class ShotFeatures:
    # Temporal
    duration: float
    relative_position: float  # shot_index / total_shots

    # Motion (mean, std, peak)
    motion_mean: float
    motion_std: float
    motion_peak: float
    
    # Audio (rich descriptors)
    rms_energy: float
    rms_delta: float
    spectral_flux: float
    pitch_mean: float
    pitch_std: float
    silence_ratio: float
    
    # Visual change
    scene_cut_strength: float
    color_hist_delta: float
```

**Key insights**:
- **Peak vs. average**: Motion/audio peaks capture surprising events
- **Change, not magnitude**: `rms_delta`, `spectral_flux`, `color_hist_delta` matter more than raw energy
- **Silence is informative**: `silence_ratio` helps identify quiet narrative moments
- **Duration matters**: Short shots may be more important in video summaries
- **Relative position**: Shot index helps GNN learn pacing patterns

### New Helper Functions:

1. **`compute_motion_scores_array()`** - Returns all frame-level motion magnitudes
2. **`compute_audio_descriptors()`** - Extracts RMS, spectral flux, pitch, silence
3. **`compute_visual_change_features()`** - Scene cuts and color histogram changes

---

## Step 3: Updated Shot & Label Schema ✅

### Old:
```python
@dataclass
class Shot:
    start: float
    end: float
    features: ShotFeatures
    importance: float      # ← REMOVED
    rank: Optional[int] = None
```

### New:
```python
@dataclass
class Shot:
    start: float
    end: float
    features: ShotFeatures
    label: Optional[float] = None   # ← For supervised learning (0/1 or score)
    rank: Optional[int] = None      # For reference only
```

**Rationale**: Labels are assigned based on dataset-specific strategies, not a universal heuristic.

---

## Step 4: Pseudo-Labels for YouTube Videos ✅

Added `apply_pseudo_labels_youtube()` function:

```python
def apply_pseudo_labels_youtube(dataset: List[VideoDataset], top_k_ratio: float = 0.15):
    """
    Top-K shots per video → label 1, rest → label 0.
    Default: Top 15% of longest shots → label 1.
    """
```

**Strategy**:
- Extract features for ALL YouTube videos (no pre-computed importance)
- Apply pseudo-labels after extraction using duration heuristic
- This aligns with F1@K evaluation metrics

---

## Step 5: Updated Process Pipeline ✅

### `process_video()`:
- Detects shots
- Extracts **rich features** (no importance scoring)
- Returns shots with `label=None` (to be assigned later)

### `build_dataset()`:
- Processes all videos
- **Before returning**: calls `apply_pseudo_labels_youtube()` to assign labels
- Saves dataset with labels in JSON

---

## Step 6: Updated Validation & Reporting ✅

### `validate_dataset()`:
- ✅ Computes label statistics instead of importance stats
- Shows: labeled count, unlabeled count, label distribution (0/1 counts)

### `test_dataset()`:
- ✅ Checks for feature descriptors instead of importance bounds
- Test 6: Verifies all shots have `motion_mean`, `rms_energy`, etc.
- Test 7: Reports label distribution

### Statistics Output:
```
📌 Labels:
   Labeled shots: 245
   Unlabeled shots: 0
   Label 0 (negative): 208
   Label 1 (positive): 37
```

---

## Benefits

1. **No More Bias**: GNN learns what matters from data, not hardcoded weights
2. **Rich Feature Space**: 13+ dimensions capture motion dynamics, audio change, visual transitions
3. **Scalable**: Works with TVSum/SumMe (real labels) + YouTube (pseudo-labels)
4. **Interpretable**: Each feature has clear meaning (duration, change, peak, variance)
5. **Flexible Labels**: Easy to apply different labeling strategies (duration, position, etc.)

---

## Usage

### Process YouTube videos with pseudo-labels:
```bash
python youtube_dataset.py --include-curated
```

This will:
1. Download/extract videos
2. Detect shots
3. Extract 13+ rich features
4. Apply Top-15% pseudo-labels
5. Save with labels in JSON

### Output Structure:
```
model/data/processed/features/
├── complete_dataset.json          # All videos with labels & features
├── features/
│   ├── music/
│   │   └── video_id_features.json  # Shots with [start, end, label, features]
│   └── ...
├── metadata/
│   └── dataset_metadata.json
└── splits/
    └── train_val_test_split.json
```

---

## Next Steps

1. **Integrate with GNN**: Pass feature vectors to graph neural network
2. **TVSum/SumMe**: Load their original importance scores as labels (normalize per-video)
3. **Training**: Use pseudo-labels + real labels as supervision signal
4. **Evaluation**: F1@K metrics on test sets

---

## File Changes Summary

| File | Changes |
|------|---------|
| `youtube_dataset.py` | ✅ Complete refactor: 1340 lines, new features, pseudo-labels |
| `ShotFeatures` | 5 fields → 14 fields |
| `Shot` | `.importance` → `.label` |
| `process_video()` | No importance computation |
| `build_dataset()` | Applies pseudo-labels before return |
| `validate_dataset()` | Label-based stats |
| `test_dataset()` | Feature descriptor checks |

---

## Key Design Decision

**Separation of Concerns**:
- **Feature Extraction Module** (this): Computes descriptors
- **Pseudo-Labeling** (`apply_pseudo_labels_youtube()`): Assigns labels based on strategy
- **GNN Model**: Learns importance from features + labels

This keeps the feature extraction clean and the labeling strategy pluggable.
