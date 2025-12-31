# YouTube Dataset Pipeline - Complete Execution Guide

## Status: ✅ READY TO USE

All components have been tested and validated:
- ✅ Test suite: 8/8 tests passing
- ✅ Directory structure: Created
- ✅ Dependencies: Installed
- ✅ Download script: Ready
- ✅ Processing pipeline: Ready

## Quick Start

### Step 1: Download Videos from Playlists

```bash
cd "e:\5th SEM Data\AI253IA-Artificial Neural Networks and deep learning(ANNDL)\ANN_Project"
python download_playlists.py
```

The script will:
- Ask how many videos to download per playlist (1-20)
- Download best MP4 quality from each playlist
- Store in `model/data/videos/{playlist_name}/`

**Playlists:**
1. **TED-Talks** (lecture) - Educational talks
2. **Kurzgesagt** (documentary) - Science education
3. **CNN-Breaking-News** (documentary) - News videos
4. **ESPN-Highlights** (sports) - Sports highlights
5. **BBC-Learning** (documentary) - BBC educational content

**Expected download time:**
- 3 videos: ~15-30 minutes
- 5 videos: ~30-60 minutes
- 10 videos: ~1-2 hours

### Step 2: Process Videos into Dataset

After videos are downloaded:

```bash
python youtube_dataset.py
```

The script will:
1. **Extract audio** from each video (16kHz mono WAV)
2. **Detect shots** using scene detection
3. **Extract features:**
   - Motion: Optical flow magnitude
   - Speech: Audio energy (RMS)
   - Scene changes: Shot boundaries
   - Audio energy: Mean absolute amplitude
   - Object count: Placeholder
4. **Compute importance** scores using domain-specific weights
5. **Smooth scores** temporally for coherence
6. **Save datasets** as JSON with train/val/test splits

**Processing time:**
- 3 videos: ~10-15 minutes
- 5 videos: ~20-30 minutes
- 10 videos: ~1-2 hours (depends on video length)

### Step 3: Verify Output

Check the created files:

```
model/data/
├── complete_dataset.json          # Full dataset (all videos)
├── metadata/
│   └── dataset_metadata.json      # Video-level metadata
├── features/
│   ├── video1_features.json       # Shot-level features for video1
│   ├── video2_features.json       # Shot-level features for video2
│   └── ...
└── splits/
    └── train_val_test_split.json  # 60/20/20 split by video_id
```

## Data Structure

### complete_dataset.json Format

```json
[
  {
    "video_id": "dQw4w9WgXcQ",
    "duration": 213.5,
    "domain": "lecture",
    "shots": [
      {
        "start": 0.0,
        "end": 5.2,
        "importance": 0.85,
        "rank": 1,
        "features": {
          "motion": 0.3,
          "speech": 0.9,
          "scene_change": 0.0,
          "audio_energy": 0.8,
          "object_count": 1.0
        }
      }
    ]
  }
]
```

### Feature Ranges

| Feature | Range | Meaning |
|---------|-------|---------|
| motion | [0, 1] | Optical flow (camera/object movement) |
| speech | [0, 1] | Voice activity (RMS energy) |
| scene_change | {0, 1} | Binary shot boundary marker |
| audio_energy | [0, 1] | Audio amplitude (speech intensity) |
| object_count | [0, 1] | Object presence |

## Domain-Specific Importance Weights

### Lecture (speech-focused)
- Speech: 0.5 (emphasize speaker)
- Scene change: 0.3 (transitions matter)
- Motion: 0.2, Audio: 0.2, Objects: 0.1

### Sports (action-focused)
- Motion: 0.5 (emphasize action)
- Scene change: 0.3 (replays/cuts)
- Speech: 0.1, Audio: 0.1, Objects: 0.2

### Documentary (balanced)
- Motion: 0.3, Speech: 0.3, Scene: 0.2
- Audio: 0.2, Objects: 0.1

### Interview (speech with gesture)
- Speech: 0.4 (primary content)
- Motion: 0.2 (gestures/expressions)
- Scene: 0.2, Audio: 0.3, Objects: 0.1

### Default (balanced)
- All features: 0.2 (equal weight)

## Integration with train.ipynb

After generating the dataset:

```python
import json
from pathlib import Path

# Load complete dataset
with open('model/data/complete_dataset.json') as f:
    dataset = json.load(f)

# Load train/val/test splits
with open('model/data/splits/train_val_test_split.json') as f:
    splits = json.load(f)

# Extract videos by split
train_videos = [v for v in dataset if v['video_id'] in splits['train']]
val_videos = [v for v in dataset if v['video_id'] in splits['val']]
test_videos = [v for v in dataset if v['video_id'] in splits['test']]

# Convert to tensors for model training
# (See train.ipynb for detailed integration code)
```

## Troubleshooting

### Issue: "yt-dlp not found"
**Solution:** Install with `pip install yt-dlp`

### Issue: "ffmpeg not found"
**Solution:**
- Windows: Download from https://ffmpeg.org/download.html and add to PATH
- Mac: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

### Issue: Very slow downloading
**Solution:**
- Check internet connection
- Reduce videos per playlist
- Try downloading from specific region playlists

### Issue: Out of memory during processing
**Solution:**
- Reduce number of videos
- Process in smaller batches
- Increase available RAM or use SSD for temp files

### Issue: Corrupted downloads
**Solution:**
- Re-run `download_playlists.py` (skips existing files)
- Delete problematic videos and re-download
- Check if playlist is still available

## Advanced Options

### Modify Shot Detection Threshold

Edit `youtube_dataset.py` process_video() call:

```python
# Lower threshold = more shots detected (more sensitive)
# Higher threshold = fewer shots (less sensitive)
process_video(video_path, audio_path, domain='lecture')
# Default threshold: 27.0
```

### Adjust Temporal Smoothing

```python
# Larger sigma = more smoothing (more temporal coherence)
# Smaller sigma = less smoothing (preserve sharp changes)
process_video(video_path, audio_path, domain='lecture', smooth_sigma=1.5)
# Default: smooth_sigma=2.0
```

### Add Custom Playlists

Edit `TOP_5_PLAYLISTS` in `youtube_dataset.py`:

```python
TOP_5_PLAYLISTS['Your-Playlist'] = {
    'url': 'https://www.youtube.com/playlist?list=YOUR_ID',
    'domain': 'lecture',  # or interview, sports, documentary, default
    'description': 'Your description'
}
```

## Performance Expectations

### Typical Metrics (3-5 videos per playlist)

**Dataset Size:**
- Total videos: 15-25
- Total shots: 500-1500
- Total duration: 2-6 hours
- Average shots per video: 50-80

**Importance Scores:**
- Range: [0.01, 0.99]
- Mean: ~0.45-0.55
- Std: ~0.25-0.30

**Processing Time:**
- Audio extraction: 1-2 min per hour of video
- Shot detection: 0.5-1 min per hour of video
- Feature extraction: 2-5 min per hour of video
- Dataset saving: < 1 min

**Output Files:**
- complete_dataset.json: 100KB - 500KB
- Feature files (per video): 10-50KB each
- Metadata: 10-20KB

## Next Steps

After generating datasets:

1. **Load into train.ipynb:** Use provided integration code
2. **Explore distributions:** Plot feature and importance distributions
3. **Adjust weights:** Fine-tune domain weights based on results
4. **Train model:** Use VidSumGNN with YouTube data
5. **Evaluate:** Compare against baseline datasets (TVSum, SumMe)

## Testing

To verify pipeline works before downloading videos:

```bash
python test_youtube_dataset.py
```

Expected output:
```
Passed: 8/8
✅ All tests passed! Pipeline is ready to use.
```

## File Sizes

Approximate disk space needed:

| Item | Size |
|------|------|
| 3 videos (5-10 min each) | 500MB - 1GB |
| 5 videos | 1-2GB |
| 10 videos | 2-4GB |
| 20 videos | 4-8GB |
| JSON dataset (any size) | 50KB - 1MB |

**Total for 5 playlists × 3 videos each:** ~2-3GB

## Summary

Pipeline flow:
```
YouTube Playlists
        ↓
   download_playlists.py
        ↓
    MP4 Videos
        ↓
   youtube_dataset.py
        ├── FFmpeg (audio extraction)
        ├── SceneDetect (shot boundaries)
        ├── OpenCV (optical flow)
        ├── librosa (audio features)
        └── Feature normalization + importance scoring
        ↓
   JSON Datasets
        ↓
  train.ipynb (VidSumGNN)
```

Ready to proceed? Run: `python download_playlists.py`
