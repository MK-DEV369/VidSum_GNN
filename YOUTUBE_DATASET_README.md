# YouTube Dataset Builder - Complete Guide

## Overview
This guide explains how to use the enhanced `youtube_dataset.py` script to automatically:
1. Download videos from top 5 major YouTube playlists
2. Extract audio and detect shots
3. Compute visual, audio, and motion features
4. Assign importance scores using domain-specific weights
5. Store datasets in organized JSON format with train/val/test splits

## Quick Start

### Step 1: Install Dependencies

```bash
pip install yt-dlp librosa opencv-python scenedetect scipy numpy
```

For FFmpeg (required for audio extraction):
- **Windows:** `choco install ffmpeg` or download from https://ffmpeg.org/download.html
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`

### Step 2: Download Videos from Playlists

The script includes 5 pre-configured playlists:

```bash
# TED Talks (lecture domain)
yt-dlp -f "best[ext=mp4]" -o "model/data/videos/TED-Talks/%(id)s.%(ext)s" \
  "https://www.youtube.com/playlist?list=PL8dPuaLjXOMNHlyLc9N7PdLQwIrq6CAOU"

# Kurzgesagt (documentary)
yt-dlp -f "best[ext=mp4]" -o "model/data/videos/Kurzgesagt/%(id)s.%(ext)s" \
  "https://www.youtube.com/playlist?list=PLFs4vir_WsTwL5P4VR_qUPp_8GwO2Oe4t"

# CNN Breaking News (documentary)
yt-dlp -f "best[ext=mp4]" -o "model/data/videos/CNN-Breaking-News/%(id)s.%(ext)s" \
  "https://www.youtube.com/playlist?list=PLRdw3IjKY2EnqDxwFHXXGzCswVFqVelPk"

# ESPN Highlights (sports)
yt-dlp -f "best[ext=mp4]" -o "model/data/videos/ESPN-Highlights/%(id)s.%(ext)s" \
  "https://www.youtube.com/playlist?list=PLRdw3IjKY2EnqDxwFHXXGzCswVFqVelPk"

# BBC Learning (documentary)
yt-dlp -f "best[ext=mp4]" -o "model/data/videos/BBC-Learning/%(id)s.%(ext)s" \
  "https://www.youtube.com/playlist?list=PL7F3y6aBeJn5UwMKM5Tov7p0F0uY_ZbQK"
```

**Or use this automated script:**

```bash
python -c "
import yt_dlp
from youtube_dataset import TOP_5_PLAYLISTS
from pathlib import Path

for name, config in TOP_5_PLAYLISTS.items():
    output_dir = Path('model/data/videos') / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f'Downloading {name}...')
        ydl.download([config['url']])
"
```

### Step 3: Run Dataset Builder

```bash
cd e:\5th\ SEM\ Data\AI253IA-Artificial\ Neural\ Networks\ and\ deep\ learning\(ANNDL\)\ANN_Project
python youtube_dataset.py
```

**Output structure:**
```
model/data/
├── videos/
│   ├── TED-Talks/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   ├── Kurzgesagt/
│   ├── CNN-Breaking-News/
│   ├── ESPN-Highlights/
│   └── BBC-Learning/
├── complete_dataset.json          # Full dataset
├── metadata/
│   └── dataset_metadata.json      # Video metadata
├── features/
│   ├── video1_features.json
│   ├── video2_features.json
│   └── ...
└── splits/
    └── train_val_test_split.json  # 60/20/20 split
```

## Dataset Structure

### complete_dataset.json
```json
[
  {
    "video_id": "abc123",
    "duration": 245.5,
    "domain": "lecture",
    "shots": [
      {
        "start": 0.0,
        "end": 10.5,
        "importance": 0.75,
        "rank": 1,
        "features": {
          "motion": 0.3,
          "speech": 0.8,
          "scene_change": 0.0,
          "audio_energy": 0.6,
          "object_count": 1.0
        }
      }
    ]
  }
]
```

### Features Explanation

| Feature | Range | Meaning |
|---------|-------|---------|
| `motion` | [0, 1] | Optical flow magnitude (camera/object movement) |
| `speech` | [0, 1] | Voice activity (RMS energy from audio) |
| `scene_change` | {0, 1} | Shot boundary (1.0 if new scene detected) |
| `audio_energy` | [0, 1] | Audio amplitude (speech intensity) |
| `object_count` | [0, 1] | Object presence (placeholder, can be enhanced) |

### Importance Score

Importance is computed as a **weighted average** of features, with domain-specific weights:

```
importance = w_motion × motion + w_speech × speech + w_scene × scene_change + ...
```

**Domain Weights:**
- **Lecture:** speech=0.5, scene=0.3 (emphasize speaker and transitions)
- **Sports:** motion=0.5, scene=0.3 (emphasize action and replays)
- **Documentary:** motion=0.3, speech=0.3, scene=0.2 (balanced)
- **Interview:** speech=0.4, motion=0.2 (speech-focused)
- **Default:** 0.2 for all features (balanced)

Scores are smoothed temporally using **Gaussian filter** (σ=2.0) for coherent summaries.

## Configuration

### Modify Top 5 Playlists

Edit `youtube_dataset.py` to change playlists:

```python
TOP_5_PLAYLISTS = {
    'Your-Playlist-Name': {
        'url': 'https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID',
        'domain': 'lecture',  # Choose: lecture, interview, sports, documentary, default
        'description': 'Your description'
    },
    # ... more playlists
}
```

### Adjust Feature Extraction Parameters

```python
# In process_video()
process_video(
    video_path,
    audio_path,
    domain='lecture',
    smooth_sigma=2.0  # Gaussian smoothing factor (larger = more temporal coherence)
)
```

### Shot Detection Threshold

```python
# In detect_shots()
detect_shots(
    video_path,
    threshold=27.0  # SceneDetect sensitivity (lower = more shots detected)
)
```

## Testing & Validation

The script automatically validates:

✓ Non-empty dataset  
✓ Video structure completeness  
✓ Shot temporal validity  
✓ Importance range [0, 1]  
✓ Domain label validity  
✓ Feature statistics  

All tests are run before saving to ensure data quality.

## Expected Output

```
======================================================================
📺 YOUTUBE VIDEO SUMMARIZATION DATASET BUILDER
======================================================================

======================================================================
Processing: TED-Talks
======================================================================
URL: https://www.youtube.com/playlist?list=PL8dPuaLjXOMNHlyLc9N7PdLQwIrq6CAOU
Domain: lecture
Description: TED Talks curated playlist

Found 523 videos
...

✓ TED-Talks: 2 videos processed

======================================================================
DATASET VALIDATION & STORAGE
======================================================================

🧪 DATASET TESTING
======================================================================
✓ Test 1: Dataset is non-empty
✓ Test 2: All videos have required fields
✓ Test 3: All shots have valid structure and importance in [0,1]
✓ Test 4: All shots have valid temporal boundaries
✓ Test 5: All videos have valid domain labels
✓ Test 6: Dataset statistics computed
   - Videos: 10
   - Total shots: 523
   - Avg shots/video: 52.3
   - Duration: 125.5h

TESTS PASSED: 6/6
======================================================================

💾 Saving dataset structure...

✓ Dataset saved successfully!
   dataset: model/data/complete_dataset.json
   metadata: model/data/metadata/dataset_metadata.json
   features_dir: model/data/features
   splits: model/data/splits/train_val_test_split.json

======================================================================
📊 FINAL DATASET STATISTICS
======================================================================
Videos: 10
Total shots: 523
Total duration: 125.5 hours
Avg shots/video: 52.3
Avg duration/video: 750.0s
Domains: lecture, documentary, sports
Importance range: [0.010, 0.995]
Mean importance: 0.487 ± 0.289
======================================================================
```

## Integration with train.ipynb

### Load the dataset:
```python
import json
from pathlib import Path

# Load complete dataset
with open('model/data/complete_dataset.json') as f:
    dataset = json.load(f)

# Load train/val/test splits
with open('model/data/splits/train_val_test_split.json') as f:
    splits = json.load(f)

# Use in training
train_videos = [v for v in dataset if v['video_id'] in splits['train']]
val_videos = [v for v in dataset if v['video_id'] in splits['val']]
test_videos = [v for v in dataset if v['video_id'] in splits['test']]
```

## Troubleshooting

### Issue: FFmpeg not found
**Solution:** Install FFmpeg globally
- Windows: Add FFmpeg to PATH
- Mac: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`

### Issue: No videos downloaded
**Solution:** Check playlist URL and internet connection
```bash
yt-dlp --list-subs "PLAYLIST_URL"  # List available formats
```

### Issue: Out of memory during feature extraction
**Solution:** Process fewer videos or increase frame skip
```python
# In compute_motion_score(), skip every N frames
frame_skip = 2  # Process every 2nd frame
```

### Issue: Very slow shot detection
**Solution:** Increase detection threshold
```python
detect_shots(video_path, threshold=50.0)  # Higher = fewer shots, faster
```

## Performance Tips

- **Batch processing:** Process multiple videos in parallel using `concurrent.futures`
- **Caching:** Save extracted audio to avoid re-extraction
- **GPU acceleration:** Use OpenCV's GPU modules for optical flow
- **Dimensionality:** Reduce frame size for faster optical flow computation

## Citation & License

Dataset format compatible with:
- TVSum [Song et al., 2015]
- SumMe [Gygli et al., 2014]
- CoSum [El-Nouby et al., 2018]

Python libraries used:
- yt-dlp: https://github.com/yt-dlp/yt-dlp
- librosa: https://librosa.org/
- OpenCV: https://opencv.org/
- PySceneDetect: https://github.com/Breakthrough/PySceneDetect
