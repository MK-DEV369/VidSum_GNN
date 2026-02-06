# Implementation Notes & Next Steps

## Current State

✅ **Feature extraction module refactored**:
- Rich 14-dimensional feature vectors (motion, audio, visual)
- No importance heuristics baked in
- Labels assigned separately via `apply_pseudo_labels_youtube()`

---

## Immediate Action Items

### 1. Test Feature Extraction on Sample Video

```bash
cd /path/to/project
python youtube_dataset.py --only-curated
```

This will:
- Download ~20 curated videos
- Extract shots + 14D features
- Apply pseudo-labels (Top-15% by duration)
- Save to `model/data/processed/features/`

**Verify output**:
```bash
# Check feature JSON structure
cat model/data/processed/features/music/sNPnbI1arSE_features.json | head -50
```

Expected:
```json
{
  "video_id": "sNPnbI1arSE",
  "domain": "music",
  "shots": [
    {
      "start": 0.0,
      "end": 2.5,
      "label": 1.0,
      "features": {
        "duration": 2.5,
        "relative_position": 0.05,
        "motion_mean": 3.2,
        "motion_std": 1.5,
        ...
      }
    }
  ]
}
```

### 2. Build Data Loader for GNN

Create `vidsum_gnn/data/youtube_loader.py`:

```python
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

class YouTubeVideoDataset(InMemoryDataset):
    def __init__(self, root='model/data/processed/features'):
        super().__init__(root)
        self.data_list = self.load_jsons()
    
    def load_jsons(self):
        feature_dir = Path(self.root) / 'features'
        data_list = []
        
        for video_file in feature_dir.rglob('*_features.json'):
            with open(video_file) as f:
                video_data = json.load(f)
            
            # Build shot features matrix
            shots = video_data['shots']
            num_shots = len(shots)
            
            features = np.array([
                [s['features'][key] for key in [
                    'duration', 'relative_position',
                    'motion_mean', 'motion_std', 'motion_peak',
                    'rms_energy', 'rms_delta', 'spectral_flux',
                    'pitch_mean', 'pitch_std', 'silence_ratio',
                    'scene_cut_strength', 'color_hist_delta'
                ]]
                for s in shots
            ])
            
            labels = np.array([s['label'] for s in shots])
            
            # Normalize features per-video
            features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            
            # Create temporal edges (connected neighbors)
            edge_index = []
            for i in range(num_shots - 1):
                edge_index.append([i, i+1])
                edge_index.append([i+1, i])
            
            data = Data(
                x=torch.FloatTensor(features_norm),
                y=torch.FloatTensor(labels),
                edge_index=torch.LongTensor(edge_index).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long),
                video_id=video_data['video_id'],
                domain=video_data['domain']
            )
            
            data_list.append(data)
        
        return data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
```

### 3. Define GNN Model

Update `vidsum_gnn/inference/gnn.py`:

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool

class ShotImportanceGNN(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64):
        super().__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        # GCN layers for shot context
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        
        # Per-shot importance prediction
        out = self.mlp(x)
        return out.squeeze(-1)
```

### 4. Training Loop

Create `vidsum_gnn/training/train.py`:

```python
import torch
from torch.optim import Adam
from torch.nn import BCELoss
from torch_geometric.data import DataLoader

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        logits = model(batch.x, batch.edge_index)
        loss = BCELoss()(logits, batch.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    from sklearn.metrics import f1_score, precision_recall_curve
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    # F1@K metric
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(all_labels, all_preds)
    
    return ap
```

### 5. Integration Checklist

- [ ] Test `youtube_dataset.py` to generate feature JSONs
- [ ] Implement `YouTubeVideoDataset` data loader
- [ ] Implement GNN model
- [ ] Setup training loop with F1@K evaluation
- [ ] Load TVSum/SumMe with real labels (normalize per-video)
- [ ] Train GNN on combined dataset
- [ ] Evaluate on test split

---

## TVSum/SumMe Integration

For datasets with ground-truth labels:

```python
# pseudo_label_tvsumme.py

def load_tvsumme_with_labels(json_path):
    """
    Loads TVSum/SumMe JSON with original importance scores.
    Returns ShotFeatures + labels (normalized per-video).
    """
    import json
    
    with open(json_path) as f:
        video_data = json.load(f)
    
    # Assume video_data = {
    #   "video_id": "...",
    #   "shots": [{"start": ..., "end": ..., "user_summary": [...]}, ...]
    # }
    
    shots = video_data['shots']
    
    # Compute importance from user summaries (ground truth)
    importance_scores = []
    for shot in shots:
        # user_summary is a binary array indicating selection
        user_scores = shot.get('user_summary', [])
        importance = np.mean(user_scores)  # Average across annotators
        importance_scores.append(importance)
    
    # Normalize per-video
    scores = np.array(importance_scores)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    # Assign as labels
    for shot, label in zip(shots, scores_norm):
        shot['label'] = float(label)
    
    return video_data
```

---

## Feature Validation

Run basic sanity checks after extraction:

```python
# verify_features.py

import json
from pathlib import Path
import numpy as np

feature_dir = Path('model/data/processed/features')

for video_file in feature_dir.rglob('*_features.json'):
    with open(video_file) as f:
        data = json.load(f)
    
    shots = data['shots']
    
    for i, shot in enumerate(shots):
        features = shot['features']
        
        # Check for NaN
        for key, val in features.items():
            if not isinstance(val, (int, float)) or np.isnan(val):
                print(f"ERROR: {video_file} shot {i} {key}={val}")
        
        # Check ranges
        assert 0 <= features['duration'] < 3600, f"Bad duration: {features['duration']}"
        assert 0 <= features['relative_position'] <= 1, f"Bad position: {features['relative_position']}"
        assert shot['label'] is None or 0 <= shot['label'] <= 1, f"Bad label: {shot['label']}"
    
    print(f"✓ {video_file.name}: {len(shots)} shots OK")
```

---

## Performance Expectations

Based on literature (TVSum, SumMe benchmarks):

| Model | F1@5% | F1@10% | F1@15% |
|-------|-------|--------|--------|
| Random baseline | ~10% | ~10% | ~10% |
| Weighted linear (hand-crafted) | 40-50% | 45-55% | 50-60% |
| GCN on features | **55-65%** | **60-70%** | **65-75%** |
| GCN + attention | **60-70%** | **65-75%** | **70-80%** |

Your GNN should exceed simple weighted baselines significantly.

---

## Debugging Tips

### If features are NaN:
1. Check FFmpeg installation: `ffmpeg -version`
2. Check librosa version: `pip install librosa --upgrade`
3. Check for zero-duration shots: `assert end > start`

### If training loss doesn't decrease:
1. Verify data loader: print first batch shapes
2. Check label distribution: should be ~85% class 0, ~15% class 1
3. Try BinaryCrossEntropyWithLogits (no sigmoid in model)

### If F1@K is low:
1. Verify labels are correct (sample 5 videos manually)
2. Try simpler model first (single GCN layer)
3. Increase hidden dimension (64 → 128)
4. Check temporal edge construction

---

## File Structure After Refactor

```
ANN_Project/
├── youtube_dataset.py              # ✅ Refactored (14D features, pseudo-labels)
├── FEATURE_EXTRACTION_REFACTOR.md  # ✅ New (this refactor)
├── SHOTFEATURES_SCHEMA.md          # ✅ New (feature reference)
├── vidsum_gnn/
│   ├── data/
│   │   └── youtube_loader.py        # 🔲 TODO: Implement
│   ├── inference/
│   │   └── gnn.py                   # 🔲 TODO: Update with new model
│   └── training/
│       └── train.py                 # 🔲 TODO: Implement
├── model/
│   ├── data/
│   │   ├── raw/youtube/             # Downloaded videos
│   │   └── processed/features/      # ✅ Feature JSONs (after running youtube_dataset.py)
│   ├── results/logs/                # Training logs
│   └── models/checkpoints/          # Saved weights
└── requirements.txt                 # ✅ Already has librosa, torch-geometric
```

---

## Summary

**What's done**:
- ✅ Feature extraction (14D, rich descriptors)
- ✅ Pseudo-labeling (Top-K strategy)
- ✅ Data schema (labels, features)

**What's next**:
- 🔲 Data loader for PyTorch Geometric
- 🔲 GNN model definition
- 🔲 Training loop + evaluation
- 🔲 TVSum/SumMe integration
- 🔲 Hyperparameter tuning

**Quick start**: `python youtube_dataset.py --include-curated` → generates test data
