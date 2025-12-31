# YouTube Dataset Builder - Complete Documentation Index

## 📍 Quick Navigation

### 🚀 **Getting Started (Start Here)**
1. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Overview of what was built
2. [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) - Step-by-step execution instructions
3. [YOUTUBE_DATASET_README.md](YOUTUBE_DATASET_README.md) - Detailed usage guide

### 🔧 **Core Components**

#### Main Pipeline
- **youtube_dataset.py** - Complete video processing pipeline
  - Downloads management and configuration
  - Audio extraction (FFmpeg)
  - Shot detection (PySceneDetect)
  - Feature extraction (OpenCV, librosa)
  - Importance scoring (domain-weighted)
  - JSON dataset generation

#### Testing & Validation
- **test_youtube_dataset.py** - Comprehensive test suite (8/8 passing)
  - Data structure validation
  - Feature normalization
  - Importance scoring
  - Temporal smoothing
  - Rank assignment
  - Dataset validation
  - Directory structure
  - Playlist configuration

#### Download Tools
- **download_playlists.py** - Interactive playlist downloader
  - Uses yt-dlp for reliable downloads
  - Supports 1-20 videos per playlist
  - Progress tracking
  - Error handling

### 📊 **The 5 YouTube Playlists**

| # | Playlist | Domain | Focus | Importance Weights |
|---|----------|--------|-------|-------------------|
| 1 | **TED-Talks** | Lecture | Educational talks | Speech: 0.5, Scene: 0.3 |
| 2 | **Kurzgesagt** | Documentary | Science education | Motion: 0.3, Speech: 0.3 |
| 3 | **CNN-Breaking-News** | Documentary | News coverage | Motion: 0.3, Speech: 0.3 |
| 4 | **ESPN-Highlights** | Sports | Sports highlights | Motion: 0.5, Scene: 0.3 |
| 5 | **BBC-Learning** | Documentary | BBC educational | Motion: 0.3, Speech: 0.3 |

**Automatic domain-based importance weighting ensures each video type is processed correctly**

---

## 📁 Directory Structure

```
ANN_Project/
├── youtube_dataset.py              ← Main pipeline (ready to use)
├── test_youtube_dataset.py         ← Test suite (all passing)
├── download_playlists.py           ← Download script (ready)
├── run_pipeline.bat                ← Complete automation (Windows)
│
├── Documentation/
│   ├── IMPLEMENTATION_SUMMARY.md   ← Project overview
│   ├── EXECUTION_GUIDE.md          ← Step-by-step guide
│   ├── YOUTUBE_DATASET_README.md   ← Detailed documentation
│   └── YOUTUBE_DATASET_INDEX.md    ← This file
│
├── Setup Scripts/
│   ├── setup_youtube_dataset.bat   ← Windows setup
│   └── setup_youtube_dataset.sh    ← Linux/Mac setup
│
├── model/
│   ├── train.ipynb                 ← VidSumGNN model
│   └── data/
│       ├── videos/                 ← Downloaded MP4 videos
│       │   ├── TED-Talks/
│       │   ├── Kurzgesagt/
│       │   ├── CNN-Breaking-News/
│       │   ├── ESPN-Highlights/
│       │   └── BBC-Learning/
│       ├── complete_dataset.json   ← Main dataset (generated)
│       ├── metadata/               ← Video metadata (generated)
│       ├── features/               ← Per-video features (generated)
│       ├── splits/                 ← Train/val/test split (generated)
│       └── processed/              ← PyTorch graphs (for model)
│
└── frontend/                        ← Web interface (separate)
```

---

## 🎯 Workflow Overview

### Phase 1: Download (15-60 minutes)
```bash
python download_playlists.py
# Prompts for video count (1-20 per playlist)
# Downloads best MP4 quality from 5 playlists
# Output: MP4 files in model/data/videos/{playlist}/
```

### Phase 2: Process (10-30 minutes)
```bash
python youtube_dataset.py
# Extracts audio, detects shots, computes features
# Generates importance scores
# Output: JSON datasets in model/data/
```

### Phase 3: Integrate (in train.ipynb)
```python
import json
with open('model/data/complete_dataset.json') as f:
    dataset = json.load(f)
# Use dataset for training VidSumGNN
```

---

## 📊 Data Pipeline Details

### Input
- YouTube videos (MP4 format, any duration/resolution)

### Processing Steps
1. **Audio Extraction** (FFmpeg)
   - Extracts audio track
   - Converts to 16kHz mono WAV
   - Stored temporarily

2. **Shot Detection** (PySceneDetect)
   - Detects scene boundaries
   - Content-based threshold (default: 27.0)
   - Produces shot start/end timestamps

3. **Motion Feature** (OpenCV)
   - Computes optical flow
   - Averages magnitude over shot
   - Normalized to [0,1]

4. **Speech Feature** (librosa)
   - Extracts RMS energy
   - Indicates voice activity
   - Normalized to [0,1]

5. **Audio Energy** (numpy)
   - Mean absolute amplitude
   - Overall audio intensity
   - Normalized to [0,1]

6. **Scene Change** (binary)
   - 1.0 at shot boundaries
   - 0.0 within shots
   - Already in [0,1]

7. **Object Count** (placeholder)
   - Can be enhanced with object detection
   - Currently: 1.0 for all shots

8. **Importance Score**
   - Weighted average of features
   - Weights depend on domain:
     - Lecture: emphasize speech
     - Sports: emphasize motion
     - Documentary: balanced
     - Interview: speech with motion
     - Default: equal weights

9. **Temporal Smoothing**
   - Gaussian filter (σ=2.0)
   - Ensures temporal coherence
   - Prevents sharp importance jumps

10. **Rank Assignment**
    - Sorts shots by importance
    - Rank 1 = most important
    - Rank N = least important

### Output
- **complete_dataset.json** - Full dataset with all videos and shots
- **dataset_metadata.json** - Video-level statistics
- **{video_id}_features.json** - Per-video shot features
- **train_val_test_split.json** - 60/20/20 split by video_id

---

## 💾 Data Format Reference

### Video Object
```json
{
  "video_id": "dQw4w9WgXcQ",
  "duration": 213.5,
  "domain": "lecture",
  "shots": [...]
}
```

### Shot Object
```json
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
```

### Expected Stats (15-25 videos)
- Total videos: 15-25
- Total shots: 500-1500
- Duration: 2-6 hours
- Avg shots/video: 50-80
- Importance: [0.01, 0.99], mean ≈ 0.45

---

## ✅ Test Coverage

| Test | Status | Coverage |
|------|--------|----------|
| Data structures | ✅ PASS | ShotFeatures, Shot, VideoDataset |
| Feature normalization | ✅ PASS | Min-max scaling to [0,1] |
| Importance scoring | ✅ PASS | 5 domain types verified |
| Temporal smoothing | ✅ PASS | Gaussian filter variance reduction |
| Rank assignment | ✅ PASS | Correct importance-based ordering |
| Dataset validation | ✅ PASS | Statistics computation |
| Directory structure | ✅ PASS | File organization |
| Playlist config | ✅ PASS | 5 playlists with domains |

**Result: 8/8 tests passing ✅**

---

## 🚀 Quick Commands

```bash
# Setup (first time only)
pip install librosa opencv-python scenedetect scipy numpy yt-dlp

# Test pipeline
python test_youtube_dataset.py

# Download videos
python download_playlists.py

# Process videos
python youtube_dataset.py

# Automated execution (Windows)
run_pipeline.bat

# Check output
dir model\data\
```

---

## 📈 Expected Performance

### Download Performance
- Bandwidth: Depends on internet speed
- Videos per playlist: 1-20 (configurable)
- Typical: 500MB-1GB per 10 videos

### Processing Performance
- Audio extraction: 1-2 min per hour of video
- Shot detection: 0.5-1 min per hour
- Feature extraction: 2-5 min per hour
- Total: ~4-8 min per hour of video

### Storage Requirements
```
Videos: 2-4 GB (for 15-25 videos)
JSON output: 300-500 KB
Total: ~2-4 GB
```

---

## 🔗 Integration Points

### With train.ipynb
```python
# Load dataset
import json
with open('model/data/complete_dataset.json') as f:
    dataset = json.load(f)

# Load splits
with open('model/data/splits/train_val_test_split.json') as f:
    splits = json.load(f)

# Filter by split
train_videos = [v for v in dataset if v['video_id'] in splits['train']]
```

### Graph Conversion for GNN
```python
# Build temporal graphs
edges = []
features = []
labels = []

for video in dataset:
    shots = video['shots']
    for i, shot in enumerate(shots):
        # Node features
        features.append([shot['features']['motion'],
                        shot['features']['speech'],
                        shot['features']['scene_change'],
                        shot['features']['audio_energy'],
                        shot['features']['object_count']])
        # Importance labels
        labels.append(shot['importance'])
        
    # Temporal edges
    for i in range(len(shots)-1):
        edges.append([i, i+1])
```

---

## 🐛 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `yt-dlp not found` | Not installed | `pip install yt-dlp` |
| `No module named 'librosa'` | Dependencies missing | Run setup script |
| `ffmpeg not found` | Not in PATH | Add to PATH or reinstall |
| Out of memory | Too many videos | Reduce playlist size |
| Very slow | Low bandwidth | Check internet speed |

---

## 📚 Related Documentation

- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Complete project overview
- [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) - Detailed execution steps
- [YOUTUBE_DATASET_README.md](YOUTUBE_DATASET_README.md) - Configuration & advanced usage
- train.ipynb - VidSumGNN model and training code

---

## 🎓 For Model Training

Once dataset is generated and saved:

1. **Load** the complete_dataset.json
2. **Extract** features and labels per shot
3. **Build** temporal graphs with shot-to-shot edges
4. **Create** train/val/test dataloaders using provided splits
5. **Train** VidSumGNN with importance scores as labels
6. **Evaluate** using F-score and Spearman correlation

---

## ✨ Key Features

✅ Automated video download from 5 major playlists  
✅ Domain-specific importance weighting  
✅ Comprehensive feature extraction (visual + audio)  
✅ Temporal coherence through smoothing  
✅ Organized JSON output format  
✅ Train/val/test splits included  
✅ Complete test suite (8/8 passing)  
✅ Integration-ready for train.ipynb  

---

## 🎯 Next Steps

1. **Read** EXECUTION_GUIDE.md for detailed instructions
2. **Run** `python test_youtube_dataset.py` to verify setup
3. **Execute** `python download_playlists.py` to download videos
4. **Run** `python youtube_dataset.py` to generate datasets
5. **Integrate** with train.ipynb for model training

**Estimated total time: 1-3 hours for complete dataset**

---

**Status: ✅ READY FOR PRODUCTION USE**

All components tested, documented, and ready to generate YouTube-based video summarization datasets.
