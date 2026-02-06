# 🚀 Model Integration Guide - VidSum GNN

**Complete Documentation for Integrating the Trained GNN Model into the Video Summarization System**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Model Integration Points](#model-integration-points)
4. [Step-by-Step Integration](#step-by-step-integration)
5. [API Endpoints](#api-endpoints)
6. [Processing Pipeline](#processing-pipeline)
7. [Configuration & Deployment](#configuration--deployment)
8. [Testing & Validation](#testing--validation)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)

---

## 📌 Overview

### Current State
The GNN model (`VidSumGNN`) has been trained and is ready for production. The model:
- **Architecture**: Graph Attention Networks (GAT) v2 with 2 layers
- **Input**: Video shots with visual + audio embeddings
- **Output**: Importance scores for each shot (0-1)
- **Location**: `model/models/results/vidsumgnn_final.pt` (trained checkpoint)

### Integration Goal
Seamlessly integrate the trained model into the FastAPI backend to:
1. Accept video uploads via the frontend
2. Process videos (shot detection, feature extraction)
3. Build a graph representation
4. Score shots using the GNN
5. Select and assemble summaries
6. Return processed video to the user

---

## 🏗️ Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│                  Upload Video + Configure Options                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    HTTP POST /upload
                             │
          ┌──────────────────▼──────────────────┐
          │    FastAPI Backend (main.py)        │
          │  Receive & Store Video              │
          └──────────────────┬──────────────────┘
                             │
        ┌────────────────────▼─────────────────────┐
        │  1. PREPROCESSING (tasks.py)             │
        │  ├─ Probe video metadata                 │
        │  ├─ Transcode to canonical format        │
        │  └─ Store processed video                │
        └────────────────────┬─────────────────────┘
                             │
        ┌────────────────────▼─────────────────────┐
        │  2. SHOT DETECTION (processing/)         │
        │  ├─ Detect scene boundaries              │
        │  ├─ Extract frame samples for each shot  │
        │  └─ Extract audio segments              │
        └────────────────────┬─────────────────────┘
                             │
        ┌────────────────────▼─────────────────────┐
        │  3. FEATURE EXTRACTION (features/)       │
        │  ├─ Visual: ViT-B/16 embeddings (768D)  │
        │  ├─ Audio: Wav2Vec2 embeddings (768D)   │
        │  └─ Concatenate: (1536D per shot)       │
        └────────────────────┬─────────────────────┘
                             │
        ┌────────────────────▼─────────────────────┐
        │  4. GRAPH CONSTRUCTION (graph/builder.py)│
        │  ├─ Create temporal edges (sequential)   │
        │  ├─ Add semantic edges (similarity)      │
        │  └─ Generate edge attributes            │
        └────────────────────┬─────────────────────┘
                             │
        ┌────────────────────▼─────────────────────┐
        │  5. GNN INFERENCE (graph/model.py) ⭐    │
        │  ├─ Load trained VidSumGNN model         │
        │  ├─ Forward pass on graph                │
        │  └─ Get importance scores (0-1)         │
        └────────────────────┬─────────────────────┘
                             │
        ┌────────────────────▼─────────────────────┐
        │  6. SHOT SELECTION (summary/selector.py) │
        │  ├─ Apply selection strategy (greedy)    │
        │  ├─ Respect target duration              │
        │  └─ Maintain temporal order              │
        └────────────────────┬─────────────────────┘
                             │
        ┌────────────────────▼─────────────────────┐
        │  7. SUMMARY ASSEMBLY (summary/assembler) │
        │  ├─ Concatenate selected shots via FFmpeg│
        │  ├─ Encode to H.264 + AAC                │
        │  └─ Store output summary                 │
        └────────────────────┬─────────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │    Store in Database + Cache         │
          │  Summary metadata & embedding        │
          └──────────────────┬──────────────────┘
                             │
                    Notify Frontend
                  (WebSocket progress)
                             │
          ┌──────────────────▼──────────────────┐
          │    Download Summary Video            │
          │     Playback in Player              │
          └──────────────────────────────────────┘
```

---

## 🔌 Model Integration Points

### 1. **Model Loading** (`vidsum_gnn/api/tasks.py`)

```python
# Load the trained GNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gnn_model = VidSumGNN(
    in_dim=1536,              # Concatenated visual + audio
    hidden_dim=1024,
    num_heads=8,
    dropout=0.2,
    edge_dim=4
).to(device)

# Load checkpoint
checkpoint = torch.load("model/models/results/vidsumgnn_final.pt", map_location=device)
gnn_model.load_state_dict(checkpoint['model_state_dict'])
gnn_model.eval()
```

### 2. **Feature Extraction** (`vidsum_gnn/features/`)

**VisualEncoder** (`visual.py`):
- Uses ViT-B/16 (Vision Transformer)
- Input: Image frame from each shot
- Output: 768-dim embedding

**AudioEncoder** (`audio.py`):
- Uses Wav2Vec2 (pretrained on speech)
- Input: Audio segment from each shot
- Output: 768-dim embedding

**Combined**: Concatenate both → 1536-dim feature per shot

### 3. **Graph Construction** (`vidsum_gnn/graph/builder.py`)

```python
builder = GraphBuilder(
    k_sim=5,              # Top-5 similar shots
    sim_threshold=0.65,   # Semantic similarity threshold
    max_edges=20          # Max edges per node
)

# Build PyTorch Geometric graph
graph_data = builder.build_graph(shots, combined_features)
# Returns: Data(x=node_features, edge_index, edge_attr)
```

**Graph Structure**:
- **Nodes**: One per shot (video segment)
- **Edges**: Temporal (sequential) + Semantic (similarity-based)
- **Node Features**: 1536-dim (visual + audio)
- **Edge Features**: 4-dim [is_temporal, distance, similarity, audio_corr]

### 4. **GNN Inference** (`vidsum_gnn/graph/model.py`)

```python
# Forward pass
shot_scores = gnn_model(
    x=graph_data.x,           # (num_shots, 1536)
    edge_index=graph_data.edge_index,
    edge_attr=graph_data.edge_attr
)
# Output: (num_shots, 1) - importance score per shot
```

### 5. **Shot Selection** (`vidsum_gnn/summary/selector.py`)

Using GNN scores + target duration:
- **Greedy Algorithm**: Select highest-scored shots until reaching target
- **Dynamic Programming**: Optimize temporal coherence
- **Constraint**: Total duration ≤ target (e.g., 60 seconds)

### 6. **Summary Assembly** (`vidsum_gnn/summary/assembler.py`)

```bash
# FFmpeg concatenation
ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4
```

---

## 🔧 Step-by-Step Integration

### Phase 1: Model Loading & Initialization

**File**: `vidsum_gnn/api/tasks.py`

```python
import torch
from vidsum_gnn.graph.model import VidSumGNN
from vidsum_gnn.core.config import settings

# Global model cache (load once)
GNN_MODEL = None
GNN_DEVICE = None

def load_gnn_model():
    global GNN_MODEL, GNN_DEVICE
    
    if GNN_MODEL is not None:
        return GNN_MODEL
    
    GNN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize architecture
    GNN_MODEL = VidSumGNN(
        in_dim=1536,
        hidden_dim=1024,
        num_heads=8,
        dropout=0.2,
        edge_dim=4
    ).to(GNN_DEVICE)
    
    # Load trained weights
    checkpoint_path = "model/models/results/vidsumgnn_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location=GNN_DEVICE)
    
    GNN_MODEL.load_state_dict(checkpoint['model_state_dict'])
    GNN_MODEL.eval()  # Set to evaluation mode
    
    logger.info(f"✓ GNN model loaded from {checkpoint_path}")
    return GNN_MODEL

async def process_video_task(video_id: str, config: dict):
    # Load model at start of pipeline
    gnn_model = load_gnn_model()
    # ... rest of pipeline
```

### Phase 2: Feature Extraction Integration

**File**: `vidsum_gnn/api/tasks.py` (in `process_video_task`)

```python
from vidsum_gnn.features.visual import VisualEncoder
from vidsum_gnn.features.audio import AudioEncoder

# Initialize encoders
visual_encoder = VisualEncoder(device=GNN_DEVICE)
audio_encoder = AudioEncoder(device=GNN_DEVICE)

# Extract features for each shot
shot_features = []

for shot in shots:
    # Get frame from shot
    frame_path = f"{shot['frame_path']}"  # Extracted during shot detection
    visual_feat = visual_encoder.encode([frame_path])  # (1, 768)
    
    # Get audio from shot
    audio_path = f"{shot['audio_path']}"  # Extracted during preprocessing
    audio_feat = audio_encoder.encode([audio_path])   # (1, 768)
    
    # Concatenate
    combined = torch.cat([visual_feat, audio_feat], dim=1)  # (1, 1536)
    shot_features.append(combined)

# Stack all features
features_tensor = torch.cat(shot_features, dim=0)  # (num_shots, 1536)
```

### Phase 3: Graph Construction & GNN Inference

**File**: `vidsum_gnn/api/tasks.py`

```python
from vidsum_gnn.graph.builder import GraphBuilder

# Build graph
builder = GraphBuilder(k_sim=5, sim_threshold=0.65)
graph_data = builder.build_graph(shots, features_tensor)

# Move to GPU
graph_data = graph_data.to(GNN_DEVICE)

# GNN inference
with torch.no_grad():
    shot_scores = gnn_model(
        x=graph_data.x,
        edge_index=graph_data.edge_index,
        edge_attr=graph_data.edge_attr
    )  # (num_shots, 1)

# Convert to numpy for downstream
scores_np = shot_scores.cpu().numpy().flatten()

# Store scores in database
for i, shot in enumerate(shots):
    shot['importance_score'] = float(scores_np[i])
```

### Phase 4: Shot Selection

**File**: `vidsum_gnn/api/tasks.py`

```python
from vidsum_gnn.summary.selector import ShotSelector

selector = ShotSelector(strategy=config['selection_method'])  # "greedy" or "dynamic"

selected_shot_indices = selector.select_shots(
    shots=shots,
    scores=scores_np,
    target_duration=config['target_duration']  # e.g., 60 seconds
)

selected_shots = [shots[i] for i in selected_shot_indices]
```

### Phase 5: Summary Assembly

**File**: `vidsum_gnn/api/tasks.py`

```python
from vidsum_gnn.summary.assembler import assemble_summary

# Generate output filename
output_path = f"data/outputs/{video_id}_summary.mp4"

# Assemble summary video
success = await assemble_summary(
    video_path=canonical_path,
    selected_shots=selected_shots,
    output_path=output_path
)

# Store result in database
summary = Summary(
    summary_id=f"{video_id}_summary",
    video_id=video_id,
    output_path=output_path,
    duration_sec=sum(s['duration_sec'] for s in selected_shots),
    num_shots_selected=len(selected_shots),
    compression_ratio=original_duration / summary_duration
)
db.add(summary)
await db.commit()
```

---

## 📡 API Endpoints

### 1. **Upload & Process Video**

**Endpoint**: `POST /upload`

**Request**:
```json
{
  "file": "<video_file>",
  "target_duration": 60,
  "selection_method": "greedy"
}
```

**Response**:
```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Video uploaded successfully. Processing started."
}
```

### 2. **Get Processing Status**

**Endpoint**: `GET /status/{video_id}`

**Response**:
```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "preprocessing",
  "progress": 35,
  "current_stage": "shot_detection",
  "message": "Detected 25 shots, starting feature extraction..."
}
```

### 3. **Get Summary**

**Endpoint**: `GET /summary/{video_id}`

**Response**:
```json
{
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "summary_id": "550e8400-e29b-41d4-a716-446655440000_summary",
  "output_path": "data/outputs/550e8400-e29b-41d4-a716-446655440000_summary.mp4",
  "duration_sec": 65,
  "original_duration_sec": 300,
  "compression_ratio": 4.62,
  "num_shots": 12,
  "created_at": "2025-12-26T10:30:45Z"
}
```

### 4. **Download Summary**

**Endpoint**: `GET /download/{video_id}`

Returns: MP4 video file

### 5. **WebSocket Progress Stream**

**Endpoint**: `WS /ws/{video_id}`

Real-time updates:
```json
{
  "timestamp": "2025-12-26T10:30:45.123Z",
  "level": "INFO",
  "message": "Processing shot 15/28",
  "stage": "feature_extraction",
  "progress": 55
}
```

---

## 🔄 Processing Pipeline

### Complete Flow in `vidsum_gnn/api/tasks.py`

```
process_video_task(video_id, config)
│
├─ [1] PREPROCESSING
│  ├─ Load video metadata (probe_video)
│  ├─ Transcode to canonical format (transcode_video)
│  └─ Progress: 20% → 30%
│
├─ [2] SHOT DETECTION
│  ├─ Detect shot boundaries (detect_shots)
│  ├─ Extract frame samples (sample_frames_for_shots)
│  ├─ Extract audio segments (extract_audio_segment)
│  └─ Progress: 30% → 45%
│
├─ [3] FEATURE EXTRACTION
│  ├─ Encode visual features (VisualEncoder)
│  ├─ Encode audio features (AudioEncoder)
│  └─ Progress: 45% → 60%
│
├─ [4] GRAPH CONSTRUCTION
│  ├─ Build graph (GraphBuilder)
│  ├─ Add temporal & semantic edges
│  └─ Progress: 60% → 65%
│
├─ [5] GNN INFERENCE ⭐
│  ├─ Load GNN model
│  ├─ Forward pass
│  ├─ Get importance scores
│  └─ Progress: 65% → 75%
│
├─ [6] SHOT SELECTION
│  ├─ Select shots based on scores & constraints
│  └─ Progress: 75% → 80%
│
├─ [7] SUMMARY ASSEMBLY
│  ├─ Concatenate selected shots (FFmpeg)
│  ├─ Encode output
│  └─ Progress: 80% → 95%
│
└─ [8] FINALIZATION
   ├─ Store in database
   ├─ Update status to "completed"
   └─ Progress: 95% → 100%
```

---

## ⚙️ Configuration & Deployment

### 1. **Model Configuration**

**File**: `vidsum_gnn/core/config.py`

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Model paths
    GNN_MODEL_PATH: str = "model/models/results/vidsumgnn_final.pt"
    
    # Model architecture
    GNN_INPUT_DIM: int = 1536      # visual + audio
    GNN_HIDDEN_DIM: int = 1024
    GNN_NUM_HEADS: int = 8
    GNN_DROPOUT: float = 0.2
    GNN_EDGE_DIM: int = 4
    
    # Graph builder parameters
    GRAPH_K_SIM: int = 5            # Top-k similar shots
    GRAPH_SIM_THRESHOLD: float = 0.65
    GRAPH_MAX_EDGES: int = 20
    
    # Feature extractors
    VISUAL_MODEL: str = "google/vit-base-patch16-224"
    AUDIO_MODEL: str = "facebook/wav2vec2-base-960h"
    
    # Processing
    SHOT_THRESHOLD: float = 27.0    # Scene change threshold
    TARGET_FPS: int = 2             # Frames per second for sampling
    BATCH_SIZE: int = 32
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 2. **Environment Setup**

**File**: `.env` (create if not exists)

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost/vidsum

# Paths
UPLOAD_DIR=data/uploads
PROCESSED_DIR=data/processed
OUTPUT_DIR=data/outputs
TEMP_DIR=data/temp

# GPU/CPU
DEVICE=cuda  # or "cpu"

# Logging
LOG_LEVEL=INFO

# Feature extractors
VISUAL_MODEL=google/vit-base-patch16-224
AUDIO_MODEL=facebook/wav2vec2-base-960h

# Model
GNN_MODEL_PATH=model/models/results/vidsumgnn_final.pt
```

### 3. **Docker Deployment**

**File**: `Dockerfile` (backend)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Download pretrained models (avoid repeated downloads)
RUN python -c "
from transformers import ViTModel, AutoModel
ViTModel.from_pretrained('google/vit-base-patch16-224')
AutoModel.from_pretrained('facebook/wav2vec2-base-960h')
"

# Run API
CMD ["uvicorn", "vidsum_gnn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://user:password@db:5432/vidsum
      DEVICE: cuda  # Use GPU if available
    volumes:
      - ./data:/app/data
      - ./model:/app/model
    depends_on:
      - db
    
  db:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: vidsum
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### 4. **Launch Backend**

```bash
# Development
uvicorn vidsum_gnn.api.main:app --reload --port 8000

# Production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker vidsum_gnn.api.main:app
```

---

## ✅ Testing & Validation

### 1. **Unit Tests for GNN Model**

**File**: `tests/test_gnn_model.py`

```python
import torch
from vidsum_gnn.graph.model import VidSumGNN

def test_gnn_forward():
    """Test GNN forward pass"""
    model = VidSumGNN(in_dim=1536, hidden_dim=1024, num_heads=8, edge_dim=4)
    
    # Mock data
    x = torch.randn(10, 1536)  # 10 shots, 1536-dim features
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.randn(3, 4)  # 3 edges, 4-dim attributes
    
    # Forward
    scores = model(x, edge_index, edge_attr)
    
    # Assertions
    assert scores.shape == (10, 1)
    assert (scores >= 0).all() and (scores <= 1).all()  # Sigmoid output
```

### 2. **Integration Test**

**File**: `tests/test_pipeline.py`

```python
import asyncio
from vidsum_gnn.api.tasks import process_video_task

async def test_full_pipeline():
    """Test complete video processing"""
    # Use test video
    test_video_id = "test_video_123"
    test_config = {
        "target_duration": 30,
        "selection_method": "greedy"
    }
    
    # Run pipeline
    await process_video_task(test_video_id, test_config)
    
    # Verify output
    # Check database entries
    # Check output file exists
```

### 3. **Test with Sample Video**

```bash
# Run test
pytest tests/test_pipeline.py -v

# Or manually via API
curl -X POST http://localhost:8000/upload \
  -F "file=@sample_video.mp4" \
  -F "target_duration=60" \
  -F "selection_method=greedy"
```

---

## 🐛 Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms**: CUDA out of memory during GNN inference

**Solution**:
```python
# In tasks.py, process in batches
BATCH_SIZE = 8  # Reduce if needed

for batch_start in range(0, num_shots, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, num_shots)
    batch_x = graph_data.x[batch_start:batch_end]
    batch_scores = gnn_model(batch_x, edge_index, edge_attr)
```

### Issue: Model Not Found

**Error**: `FileNotFoundError: model/models/results/vidsumgnn_final.pt`

**Solution**:
```bash
# Ensure model file exists
ls -la model/models/results/

# Or download from cloud storage
python -c "from vidsum_gnn.utils.model_loader import download_pretrained; download_pretrained()"
```

### Issue: Slow Processing

**Symptoms**: Each video takes >10 minutes

**Solution**:
- Enable GPU: `DEVICE=cuda` in `.env`
- Reduce shot count: Lower `SHOT_THRESHOLD`
- Use smaller models: Switch to ViT-tiny
- Increase batch size: `BATCH_SIZE=64`

### Issue: Poor Summary Quality

**Symptoms**: Summary doesn't capture important scenes

**Solution**:
1. Re-train GNN with labeled data
2. Adjust graph builder parameters:
   ```python
   builder = GraphBuilder(k_sim=10, sim_threshold=0.5)
   ```
3. Try different selection strategies:
   - `"greedy"`: Fast, reasonable quality
   - `"dynamic"`: Slower, higher quality

---

## ⚡ Performance Optimization

### 1. **Model Inference Caching**

```python
# Cache features to avoid re-extraction
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_features(video_id: str):
    # Load from database or disk
    return features_tensor
```

### 2. **Batch Processing**

```python
# Process multiple videos in parallel
async def process_batch(video_ids: List[str]):
    tasks = [process_video_task(vid, config) for vid in video_ids]
    results = await asyncio.gather(*tasks)
    return results
```

### 3. **Checkpointing**

```python
# Save intermediate results to avoid recomputation
if os.path.exists(f"data/temp/{video_id}_features.pt"):
    features = torch.load(f"data/temp/{video_id}_features.pt")
else:
    features = extract_features(...)
    torch.save(features, f"data/temp/{video_id}_features.pt")
```

### 4. **Model Quantization** (Optional)

```python
# Convert to FP16 for faster inference
gnn_model = gnn_model.half()

# Forward pass
with torch.cuda.amp.autocast():
    scores = gnn_model(x.half(), edge_index, edge_attr.half())
```

### 5. **MultiGPU Processing** (Advanced)

```python
# Distribute across multiple GPUs
if torch.cuda.device_count() > 1:
    gnn_model = torch.nn.DataParallel(gnn_model)
```

---

## 📊 Monitoring & Logging

### Real-time Progress Tracking

**Frontend** receives WebSocket updates:
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/${videoId}`);

ws.onmessage = (event) => {
  const log = JSON.parse(event.data);
  console.log(`[${log.stage}] ${log.message} (${log.progress}%)`);
  updateProgressBar(log.progress);
};
```

### Database Logging

All processing stages logged in `Video` and `Summary` tables:
```sql
SELECT 
  video_id,
  status,
  created_at,
  updated_at,
  total_duration,
  processing_time
FROM videos
ORDER BY created_at DESC;
```

---

## 📚 Additional Resources

### Key Files Reference

| File | Purpose |
|------|---------|
| `vidsum_gnn/graph/model.py` | GNN architecture (VidSumGNN) |
| `vidsum_gnn/api/tasks.py` | Main processing pipeline |
| `vidsum_gnn/features/visual.py` | Visual feature extraction |
| `vidsum_gnn/features/audio.py` | Audio feature extraction |
| `vidsum_gnn/graph/builder.py` | Graph construction |
| `vidsum_gnn/summary/selector.py` | Shot selection strategies |
| `vidsum_gnn/summary/assembler.py` | Video assembly (FFmpeg) |
| `model/train.ipynb` | Model training notebook |

### Important Parameters

```python
# GNN Architecture
in_dim = 1536         # Feature dimension (visual=768 + audio=768)
hidden_dim = 1024     # Hidden layer size
num_heads = 8         # Attention heads
dropout = 0.2         # Dropout rate
edge_dim = 4          # Edge attribute dimension

# Graph Builder
k_sim = 5             # Top-k similar shots
sim_threshold = 0.65  # Similarity threshold
max_edges = 20        # Max edges per node

# Processing
shot_threshold = 27.0 # Scene change detection
batch_size = 32       # Processing batch size
target_duration = 60  # Default summary duration (seconds)
```

---

## 🎯 Quick Start Checklist

- [ ] Model checkpoint exists at `model/models/results/vidsumgnn_final.pt`
- [ ] Backend dependencies installed: `pip install -r requirements.txt`
- [ ] Database configured and running (PostgreSQL + TimescaleDB)
- [ ] GPU available (optional but recommended)
- [ ] Environment variables set in `.env`
- [ ] Backend API running: `uvicorn vidsum_gnn.api.main:app --reload`
- [ ] Frontend running: `cd frontend && npm run dev`
- [ ] Test upload via frontend or API
- [ ] Monitor progress in WebSocket logs
- [ ] Download summary video

---

## 📞 Support & Troubleshooting

**For issues or questions**:
1. Check logs: `data/logs/*.log`
2. Review this guide
3. Check test results: `pytest tests/`
4. Enable debug logging: `LOG_LEVEL=DEBUG` in `.env`

---

**Last Updated**: December 26, 2025  
**Model Status**: Production Ready ✅  
**Integration Status**: Complete ✅
