# UGC Dataset Analysis & Integration

## Overview

**UGC (User-Generated Content) Dataset Location**: `model/data/raw/ugc/`

Successfully integrated UGC dataset processing into `youtube_dataset.py`. The script now supports processing both YouTube playlists and UGC videos in a unified pipeline.

---

## UGC Dataset Composition

### Video Count by Domain
- **Gaming**: 35 videos (1080P & 720P variants)
- **Sports**: 55 videos (1080P & 720P variants)
- **Vlog**: 35 videos (1080P & 720P variants)
- **Total**: 125 videos

### File Naming Convention
Videos follow pattern: `{Domain}_{Resolution}-{ID}_orig.mp4`
- Examples:
  - `Gaming_1080P-0ce6_orig.mp4`
  - `Sports_720P-0b9e_orig.mp4`
  - `Vlog_1080P-010b_orig.mp4`

---

## Current Dataset Status

### YouTube Dataset (Processed ✓)
```
Playlist: TED-Talks (lecture) → ~10 videos
Playlist: Kurzgesagt (documentary) → ~10 videos
Playlist: ESPN-Highlights (sports) → 2 videos
─────────────────────────────────────────
Total: 22 videos | 940 shots | 4.61 hours
Domains: lecture, documentary, sports
```

### Complete Dataset After UGC Integration
**Expected when UGC is included**:
```
YouTube: 22 videos | 940 shots
UGC: 125 videos | ~5,000+ shots (estimated)
─────────────────────────────────────────
Total: 147 videos | ~5,900+ shots | ~25+ hours
Domains: lecture, documentary, sports, gaming, vlog
```

---

## Domain-Specific Weights (UGC)

### Gaming Domain
- **Characteristics**: High motion, varied audio, dynamic scenes
- **Weights**: `{"motion": 0.6, "speech": 0.15, "scene_change": 0.15, "audio_energy": 0.05, "object_count": 0.05}`
- **Why**: Gaming videos have rapid camera movement, sound effects, but less dialogue

### Vlog Domain
- **Characteristics**: Moderate motion, high speech, casual narration
- **Weights**: `{"motion": 0.2, "speech": 0.5, "scene_change": 0.1, "audio_energy": 0.15, "object_count": 0.05}`
- **Why**: Vlogs emphasize speaker/narrator content over visual action

### Sports Domain (Updated)
- **Keep existing weights**: `{"motion": 0.5, "speech": 0.1, "scene_change": 0.2, "audio_energy": 0.1, "object_count": 0.1}`
- **Note**: Applies to both YouTube ESPN videos and UGC sports videos

---

## Code Changes Made

### 1. Updated Constants
```python
RAW_YOUTUBE_DIR = Path("model/data/raw/youtube")
RAW_UGC_DIR = Path("model/data/raw/ugc")           # NEW
PROCESSED_OUTPUT_DIR = Path("model/data/processed/features")
```

### 2. Added UGC Domain Detector
```python
def get_ugc_domain_from_filename(filename: str) -> str:
    """Extract domain from filename: Gaming_*, Sports_*, Vlog_*"""
    if filename.startswith("Gaming_"):
        return "gaming"
    elif filename.startswith("Sports_"):
        return "sports"
    elif filename.startswith("Vlog_"):
        return "vlog"
    else:
        return "default"
```

### 3. Enhanced build_dataset()
- Added `include_ugc` parameter
- Processes UGC folder after YouTube playlists
- Auto-detects domain from filename
- Maintains consistent feature extraction and importance scoring

### 4. Updated Main Execution
- Added `--ugc` or `-u` flag support
- Updated usage instructions

---

## How to Run

### Process YouTube Only (Current Default)
```bash
python youtube_dataset.py
```

### Process YouTube + UGC Together
```bash
python youtube_dataset.py --ugc
```

### Process Specific Playlist Only
```bash
python youtube_dataset.py "ESPN-Highlights"
```

### Process Specific Playlist + UGC
```bash
python youtube_dataset.py "ESPN-Highlights" --ugc
```

---

## Output Structure

After processing, the dataset is organized as:

```
model/data/processed/features/
├── complete_dataset.json          # All videos + shots + features
├── metadata/
│   └── dataset_metadata.json      # Video-level statistics
├── features/
│   ├── {video_id}_features.json   # Per-shot features (145+ files)
│   └── ...
└── splits/
    └── train_val_test_split.json  # 60% train, 20% val, 20% test
```

---

## Data Quality Expectations

### For UGC Dataset
- **Shot Detection**: Expect variable shot counts depending on edit frequency
  - Gaming: 50-100+ shots (fast edits, scene changes)
  - Sports: 30-80+ shots (frequent cuts, replays)
  - Vlog: 20-50 shots (moderate editing)
- **Audio Quality**: Variable; some videos may have background music, narration, or gameplay audio
- **Motion Patterns**: High motion in gaming, varied in sports, moderate in vlogs

### Quality Metrics
Each shot gets validated for:
- ✓ Temporal boundary consistency (start < end)
- ✓ Importance score in range [0, 1]
- ✓ Feature completeness (motion, speech, audio_energy, etc.)
- ✓ Domain label validity

---

## Processing Time Estimate

Based on ESPN (sports, motion-heavy):

| Domain | Avg Shots/Video | Time/Video | Est. Total |
|--------|-----------------|-----------|-----------|
| Gaming | 70 | 20-30 min | 11-17 hours |
| Sports | 50 | 15-25 min | 13-23 hours |
| Vlog | 35 | 10-15 min | 5.8-8.75 hours |
| **Total UGC** | **52** | **18.3 min** | **37-50 hours** |

> **Note**: Motion computation (optical flow) is the bottleneck. Faster CPU/GPU significantly reduces time.

---

## Dataset Readiness

✅ **Complete dataset structure supports**:
- Multi-domain training (5 domains: lecture, documentary, sports, gaming, vlog)
- Domain-aware importance scoring
- Train/val/test splits (deterministic by index)
- Per-shot feature extraction for GNN training
- Metadata for evaluation and benchmarking

✅ **Ready for**:
- Training VidSumGNN on expanded dataset
- Fine-tuning with multimodal features
- Evaluation across different content types
- Cross-domain generalization studies

---

## Notes

1. **No Ground Truth**: Like YouTube dataset, UGC videos have no manual summaries. Importance scores are computed heuristically using domain weights + temporal smoothing.
2. **Domain Auto-Detection**: Filename-based detection assumes naming convention holds. Verify if videos are correctly classified.
3. **Resolution Variants**: 1080P and 720P versions exist; no quality filtering applied (use as-is).
4. **Backward Compatibility**: Complete dataset will include both YouTube (22) and UGC (125) = 147 total videos. Old YouTube-only results can be recovered by not using `--ugc` flag.

---

## Next Steps (Optional)

1. Run: `python youtube_dataset.py --ugc` to generate UGC features
2. Verify output in `model/data/processed/features/metadata/dataset_metadata.json`
3. Check domain distribution in final dataset
4. Use expanded dataset to retrain or fine-tune VidSumGNN
5. Benchmark cross-domain performance (e.g., train on YouTube, test on UGC)
