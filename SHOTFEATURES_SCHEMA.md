# ShotFeatures Schema - Complete Reference

## Feature Dimensions (14 total)

### Temporal (2)
| Feature | Type | Range | Meaning |
|---------|------|-------|---------|
| `duration` | float | [0, ∞) | Length of shot in seconds |
| `relative_position` | float | [0, 1] | Shot index / total shots in video |

### Motion (3)
| Feature | Type | Range | Meaning |
|---------|------|-------|---------|
| `motion_mean` | float | [0, ∞) | Average optical flow magnitude |
| `motion_std` | float | [0, ∞) | Std dev of optical flow (variability) |
| `motion_peak` | float | [0, ∞) | Maximum optical flow in shot |

**Insight**: Peak motion captures action moments; std measures consistency

### Audio (6)
| Feature | Type | Range | Meaning |
|---------|------|-------|---------|
| `rms_energy` | float | [0, 1] | Mean RMS energy (loudness) |
| `rms_delta` | float | [0, ∞) | Std dev of RMS (dynamic range) |
| `spectral_flux` | float | [0, ∞) | Rate of spectrum change |
| `pitch_mean` | float | [Hz] | Mean fundamental frequency |
| `pitch_std` | float | [Hz] | Pitch variability (emotion/dynamics) |
| `silence_ratio` | float | [0, 1] | Fraction of silent frames |

**Insight**: Change matters more than magnitude; silence indicates dialogue/narrative

### Visual (2)
| Feature | Type | Range | Meaning |
|---------|------|-------|---------|
| `scene_cut_strength` | float | [0, 1] | Perceptual difference: first → last frame |
| `color_hist_delta` | float | [0, ∞) | Bhattacharyya distance: color distribution |

**Insight**: Captures shot transitions (cuts, fades, color changes)

---

## Usage Example

```python
# Extract features for a shot
from youtube_dataset import extract_shot_features
from pathlib import Path

video_path = Path("video.mp4")
audio_path = Path("audio.wav")

features = extract_shot_features(
    video_path=video_path,
    audio_path=audio_path,
    start=10.0,
    end=15.0,
    domain="music",
    shot_index=3,
    total_shots=20
)

print(f"Motion peak: {features.motion_peak}")
print(f"Silence ratio: {features.silence_ratio}")
print(f"Scene change: {features.scene_cut_strength}")

# Convert to dict for ML pipeline
feature_dict = features.to_dict()
```

---

## GNN Input Format

Each shot in the dataset JSON:

```json
{
  "start": 10.5,
  "end": 15.3,
  "label": 1.0,
  "rank": 5,
  "features": {
    "duration": 4.8,
    "relative_position": 0.15,
    "motion_mean": 2.34,
    "motion_std": 1.12,
    "motion_peak": 5.67,
    "rms_energy": 0.32,
    "rms_delta": 0.08,
    "spectral_flux": 0.45,
    "pitch_mean": 120.5,
    "pitch_std": 25.3,
    "silence_ratio": 0.05,
    "scene_cut_strength": 0.25,
    "color_hist_delta": 0.18
  }
}
```

---

## Computing Features

### Feature Extraction Pipeline:

1. **Motion** (OpenCV optical flow)
   - Frame-by-frame optical flow using Farneback method
   - Compute mean, std, peak magnitudes

2. **Audio** (Librosa)
   - RMS energy from waveform
   - Spectral flux from mel-spectrogram
   - F0 (pitch) using Yin algorithm
   - Silence detection on RMS threshold

3. **Visual** (OpenCV histograms)
   - Absolute difference: first vs. last frame
   - Color histogram (8×8×8 bins) Bhattacharyya distance

---

## Normalization Strategy

Features are **NOT** normalized at extraction time. This allows:

1. **Per-video normalization** (standard scaling)
2. **Per-domain normalization** (domain-specific ranges)
3. **GNN to learn scales** (batch normalization in model)

---

## Domain Notes

### Motion-disabled domains:
- `lecture`, `interview` (low motion, prioritize audio/speech)

For these, `motion_mean = 0` but `motion_std`, `motion_peak` also zero.

---

## Pseudo-Labeling Strategy

For YouTube videos (no ground truth):

```python
apply_pseudo_labels_youtube(dataset, top_k_ratio=0.15)
```

**Assigns**:
- Top 15% shots (by duration) → `label = 1.0`
- Rest → `label = 0.0`

This can be customized:
```python
# Alternative: by motion peak
top_k = max(1, int(np.ceil(len(shots) * 0.15)))
top_indices = np.argsort([s.features.motion_peak for s in shots])[-top_k:]
```

---

## Compatibility with Existing Datasets

### TVSum/SumMe:
- Import original importance scores
- Normalize per-video: `score_norm = (score - min) / (max - min)`
- Directly assign as `.label`

### YouTube:
- Start with `label=None`
- Apply `apply_pseudo_labels_youtube()` for initial supervision
- GNN learns to refine

---

## Feature Statistics

Typical ranges (example music video):

| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| duration | 0.5s | 8.0s | 2.3s | 1.8s |
| motion_mean | 0 | 15 | 3.2 | 2.1 |
| rms_energy | 0 | 1 | 0.45 | 0.2 |
| pitch_mean | 0 | 400Hz | 150Hz | 80Hz |
| silence_ratio | 0 | 1 | 0.15 | 0.12 |

---

## Troubleshooting

### Missing audio:
- `rms_energy`, `spectral_flux`, `pitch_mean/std`, `silence_ratio` → 0.0

### Very short shots (< 50ms):
- `spectral_flux` may be unstable
- Features still valid; GNN learns to discount short shots

### High motion variance:
- `motion_std` >> `motion_mean` indicates jittery, unstable movement
- Useful signal for identifying action sequences

---

## Next Steps

1. **Training**: Feed feature vectors + labels to GNN
2. **Normalization**: Implement per-video or per-domain scaling in data loader
3. **Graph Construction**: Connect shots temporally; optionally add visual/audio edges
4. **Loss Function**: Binary cross-entropy (or regression on [0,1] scores for TVSum/SumMe)
