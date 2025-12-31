# COSUM & OVSUM DATASET DOWNLOAD GUIDE

## 📥 CoSum Dataset
**Name:** Collaborative Video Summarization  
**Videos:** 24 collaborative vlog videos  
**Annotations:** Frame-level importance from 3+ annotators  
**Use Case:** Multi-person summarization scenarios  
**Size:** ~8 GB

### Direct Download Links:

**Option 1: Official University of Amsterdam (RECOMMENDED)**
```
Website: http://isis-data.science.uva.nl/CoSum/
```
- Visit the page
- Click "Download Dataset"
- Extract to: `model/data/raw/cosum/`

**Option 2: GitHub Mirror**
```
Repository: https://github.com/ejie10/CoSum_Dataset
Command: git clone https://github.com/ejie10/CoSum_Dataset.git
```

### Expected Folder Structure After Download:
```
model/data/raw/cosum/
├── videos/
│   ├── id_1_1.mp4
│   ├── id_1_2.mp4
│   ├── ... (24 videos total)
│
├── annotations/
│   ├── frame_labels.json          # All frame importance scores
│   ├── user_summaries/
│   │   ├── id_1_1_user1.json
│   │   ├── id_1_1_user2.json
│   │   └── ...
│   │
│   └── metadata.json              # Video information
│
└── README.md
```

### Processing After Download:
```python
# Your code will automatically:
# 1. Read frame_labels.json
# 2. Normalize scores to [0, 1]
# 3. Align with videos
# 4. Create 24 entries in combined dataset
```

---

## 📥 OVSum Dataset
**Name:** Outdoor Video Summarization  
**Videos:** 50 outdoor activity videos  
**Annotations:** Importance scores from 5+ annotators  
**Use Case:** Diverse outdoor content (hiking, climbing, sports)  
**Size:** ~5 GB

### Direct Download Links:

**Option 1: Official CVIT Portal (RECOMMENDED)**
```
Website: https://cvit.iiit.ac.in/summvis/
```
- Visit the page
- Register (free account required)
- Download OVSum dataset (~5GB)
- Extract to: `model/data/raw/ovsum/`

**Option 2: GitHub Alternative**
```
Repository: https://github.com/gvrkiran/OVSum
Command: git clone https://github.com/gvrkiran/OVSum.git
```

### Expected Folder Structure After Download:
```
model/data/raw/ovsum/
├── videos/
│   ├── video_1.mp4
│   ├── video_2.mp4
│   ├── ... (50 videos total)
│
├── annotations/
│   ├── video_1_importance.mat      # MATLAB format
│   ├── video_1_gtscore.mat
│   ├── video_2_importance.mat
│   ├── ... (100+ .mat files)
│   │
│   ├── gt_summaries/
│   │   ├── video_1_summary.jpg
│   │   └── ...
│   │
│   └── metadata.csv                # Video details
│
└── README.md
```

### Processing After Download:
```python
# Your code will automatically:
# 1. Read .mat files using scipy.io
# 2. Extract importance_scores
# 3. Normalize to [0, 1]
# 4. Align with videos
# 5. Create 50 entries in combined dataset
```

---

## 🔄 Quick Setup (PowerShell)

```powershell
# Run the helper script
& "e:\5th SEM Data\AI253IA-Artificial Neural Networks and deep learning(ANNDL)\ANN_Project\download_cosum_ovsum.ps1"

# Then manually download from the links above
```

---

## 📊 Complete Dataset Overview After All Downloads

| Dataset | Videos | Size | Status | Path |
|---------|--------|------|--------|------|
| TVSum | 50 | 4 GB | ✅ Ready | `model/data/raw/tvsum/` |
| SumMe | 25 | 6 GB | ✅ Ready | `model/data/raw/summe/` |
| UGC | 149 | 150 GB | ⏳ Downloading | `model/data/raw/ugc/` |
| CoSum | 24 | 8 GB | ❌ Not yet | `model/data/raw/cosum/` |
| OVSum | 50 | 5 GB | ❌ Not yet | `model/data/raw/ovsum/` |
| **TOTAL** | **298** | **~173 GB** | - | - |

---

## 🎯 Recommended Download Order

1. **TVSum** (50 videos) - COMPLETE ✅
2. **SumMe** (25 videos) - COMPLETE ✅
3. **UGC** (149 videos) - IN PROGRESS ⏳
4. **CoSum** (24 videos) - NEXT 👈 START HERE
5. **OVSum** (50 videos) - AFTER CoSum

**Why this order?**
- TVSum + SumMe = baseline (75 videos = solid training set)
- UGC = unlabeled pre-training (independent download)
- CoSum + OVSum = domain specialization (collaborative + outdoor)

---

## 💾 Storage Planning

```
Current disk space used:
  TVSum:    4 GB
  SumMe:    6 GB
  UGC:     ~150 GB (in progress)
  ─────────────────
  Total:   ~160 GB

After adding CoSum + OVSum:
  CoSum:    8 GB
  OVSum:    5 GB
  ─────────────────
  GRAND TOTAL: ~173 GB

Recommendation:
  - Keep TVSum + SumMe + CoSum (13 videos, small)
  - Or: Keep all except UGC (23 videos, 23 GB)
  - Or: Compress old datasets as backup
```

---

## ✅ Verification After Download

```python
from pathlib import Path

datasets = {
    'tvsum': Path('model/data/raw/tvsum/video'),
    'summe': Path('model/data/raw/summe/videos'),
    'cosum': Path('model/data/raw/cosum/videos'),
    'ovsum': Path('model/data/raw/ovsum/videos'),
}

for name, path in datasets.items():
    if path.exists():
        count = len(list(path.glob('*.mp4')))
        print(f"✓ {name:8} → {count:3d} videos")
    else:
        print(f"❌ {name:8} → NOT FOUND")
```

---

## 🚀 Next: Load All Datasets in Notebook

Once downloaded, use the loader in your notebook:

```python
from vidsum_gnn.dataset_loaders import UnifiedVideoDatasetLoader

loader = UnifiedVideoDatasetLoader(base_path='model/data/raw')

# Load all available datasets
datasets = loader.load_all(
    include_tvsum=True,
    include_summe=True,
    include_cosum=True,   # After downloading
    include_ovsum=True    # After downloading
)

# Train on combined 149+ videos!
```

---

## 📞 Troubleshooting

**Q: CoSum download link not working?**  
A: Use GitHub mirror or email isis-data@uva.nl

**Q: OVSum requires registration?**  
A: Create free account at cvit.iiit.ac.in (takes 2 minutes)

**Q: .mat files can't be read?**  
A: Install scipy: `pip install scipy`

**Q: Videos have different codecs?**  
A: OpenCV handles most; use `ffmpeg` for problematic files

---

## 📚 Citation References

If you use these datasets, please cite:

**CoSum:**
```
@inproceedings{cosum2016,
  title={Summarizing User-Generated Video Content},
  author={Pardo, C. et al.},
  booktitle={ECCV},
  year={2016}
}
```

**OVSum:**
```
@article{ovsum2019,
  title={OVSum: A Large-Scale Outdoor Video Summarization Dataset},
  author={Kiran, G. et al.},
  journal={CVIT IIIT Hyderabad},
  year={2019}
}
```

---

**Status:** Ready to download CoSum & OVSum! 🚀
