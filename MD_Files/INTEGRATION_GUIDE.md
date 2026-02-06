# Local Setup & Integration Guide

## Overview
This guide walks you through setting up the complete VideoSum-GNN system with trained model integration, local database, and optimized endpoints.

## Prerequisites
- Python 3.10+
- PostgreSQL 14+ (or TimescaleDB for time-series optimization)
- Node.js 18+ and npm/yarn
- CUDA-capable GPU (recommended for faster inference)
- 16GB+ RAM

## 1. Database Setup

### Option A: PostgreSQL (Recommended for Local)
```bash
# Install PostgreSQL (Windows)
# Download from: https://www.postgresql.org/download/windows/

# Create database
createdb vidsum_gnn

# Or using psql
psql -U postgres
CREATE DATABASE vidsum_gnn;
\q
```

### Option B: Docker PostgreSQL
```bash
# Start PostgreSQL container
docker run -d \
  --name vidsum-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=vidsum_gnn \
  -p 5432:5432 \
  postgres:14-alpine

# Verify connection
docker exec -it vidsum-postgres psql -U postgres -d vidsum_gnn
```

### Configure Database URL
Update [vidsum_gnn/core/config.py](vidsum_gnn/core/config.py#L10):
```python
DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/vidsum_gnn"
```

### Initialize Schema
```bash
# Run migrations (creates tables)
cd vidsum_gnn/db
python -c "from vidsum_gnn.db.models import Base; from vidsum_gnn.db.client import engine; import asyncio; asyncio.run(Base.metadata.create_all(bind=engine))"

# Or use alembic if configured
alembic upgrade head
```

### Apply Database Optimizations
```bash
python optimize_database.py
```

## 2. Backend Setup

### Install Dependencies
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements-local.txt

# Install project in editable mode
pip install -e .
```

### Download Pre-trained Models
The system automatically downloads models on first run, but you can pre-download:

```python
# Run this to pre-download models
python -c "
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer

# Whisper base (~290MB)
WhisperProcessor.from_pretrained('openai/whisper-base')
WhisperForConditionalGeneration.from_pretrained('openai/whisper-base')

# Sentence-BERT (~80MB)
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Flan-T5 base (~990MB)
from transformers import T5Tokenizer, T5ForConditionalGeneration
T5Tokenizer.from_pretrained('google/flan-t5-base')
T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
"
```

### Verify Trained Model
Ensure your trained GNN model checkpoint exists:
```bash
ls model/models/checkpoints/best_model.pt
# Should show: model/models/checkpoints/best_model.pt
```

### Start Backend Server
```bash
cd vidsum_gnn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: http://localhost:8000
API docs: http://localhost:8000/docs

## 3. Frontend Setup

### Install Dependencies
```bash
cd frontend
npm install
# or yarn install
```

### Configure API Endpoint
Update [frontend/src/pages/DashboardPage.tsx](frontend/src/pages/DashboardPage.tsx#L25) if needed:
```typescript
const API_BASE = "http://localhost:8000";
```

### Start Frontend Dev Server
```bash
npm run dev
# or yarn dev
```

Frontend will be available at: http://localhost:5173

## 4. End-to-End Test

### Upload and Process Video
1. Navigate to http://localhost:5173
2. Select a video file (MP4, AVI, MOV)
3. Configure settings:
   - **Target Duration**: 30-60 seconds recommended
   - **Selection Method**: Greedy (faster) or Knapsack (optimal)
   - **Summary Type**: 
     - Balanced: Equal visual + audio weighting
     - Visual: Prioritizes action/scene changes
     - Audio: Prioritizes speech/music
     - Highlight: Peak moments only
4. Click "Upload & Process"
5. Monitor real-time logs in processing panel
6. View results:
   - Video summary (downloadable MP4)
   - Text summary (bullet/structured/plain formats)

### Test API Directly
```bash
# Upload video
curl -X POST http://localhost:8000/api/upload \
  -F "file=@test_video.mp4" \
  -F "target_duration=30" \
  -F "selection_method=greedy" \
  -F "summary_type=balanced"

# Response: {"video_id": "abc123", "message": "Processing started"}

# Check status
curl http://localhost:8000/api/status/abc123

# Get text summary
curl http://localhost:8000/api/summary/abc123/text

# Download summary video
curl -O http://localhost:8000/api/download/abc123
```

## 5. Model Integration Details

### Architecture Flow
```
Video Upload
    ↓
Shot Detection (PySceneDetect)
    ↓
Feature Extraction
    ├─ Visual: CLIP (768-dim)
    └─ Audio: Wav2Vec2 (768-dim)
    ↓
Graph Construction (temporal + semantic edges)
    ↓
GNN Inference (3-layer GAT)
    ├─ Input: 1536-dim (concat visual+audio)
    ├─ Hidden: 512-dim × 3 layers
    └─ Output: Importance scores per shot
    ↓
Shot Selection (Greedy/Knapsack)
    ↓
Summary Assembly (FFmpeg concatenation)
    ↓
Text Summarization
    ├─ Audio → Whisper transcription
    ├─ Top-K shots by GNN scores
    ├─ Text embedding (Sentence-BERT)
    └─ LLM summary (Flan-T5)
    ↓
Output: Video + Text (3 formats)
```

### Model Service Components
- **VidSumGNN**: Trained GAT model from [model/models/checkpoints/best_model.pt](model/models/checkpoints/best_model.pt)
- **AudioTranscriber**: Whisper base (English ASR)
- **TextEmbedder**: MiniLM-L6-v2 (384-dim embeddings)
- **LLMSummarizer**: Flan-T5 base (free, no API key required)

### Customization
Adjust model hyperparameters in [vidsum_gnn/model_service.py](vidsum_gnn/model_service.py):
```python
# Change Whisper model size
AudioTranscriber(model_name="openai/whisper-small")  # or medium, large

# Adjust summarization length
LLMSummarizer(max_length=256, min_length=64)

# Change top-K shot selection
model_service.generate_text_summary(..., top_k=10)  # default 15
```

## 6. Performance Optimization

### Backend Optimizations Applied
✓ Database indexes on `summaries(video_id, type)` and `shots(video_id, importance_score)`
✓ Async SQLAlchemy with connection pooling (20 connections)
✓ Model singleton pattern (loads once, reused)
✓ Audio transcription caching (JSON files)
✓ Text embedding caching (.npy files)

### Additional Optimizations (Optional)
```python
# 1. Enable PyTorch compilation (PyTorch 2.0+)
# In vidsum_gnn/model_service.py:
self.model = torch.compile(self.model)

# 2. Batch audio processing
# Modify process_video_end_to_end to use asyncio.gather for parallel transcription

# 3. Redis caching for completed summaries
# Add caching layer before database queries

# 4. GPU memory optimization
# Reduce batch size or use FP16 inference:
with torch.cuda.amp.autocast():
    scores = self.model(node_features, edge_index)
```

### Frontend Optimizations Applied
✓ React.memo for Card components
✓ WebSocket connection management (auto-cleanup)
✓ Lazy loading of heavy components

## 7. Production Deployment

### Docker Deployment
```bash
# Build and start all services
docker-compose up --build

# Services:
# - Backend: http://localhost:8000
# - Frontend: http://localhost:80
# - PostgreSQL: localhost:5432
```

### Environment Variables
Create `.env` file:
```env
DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/vidsum_gnn
UPLOAD_DIR=/app/data/uploads
PROCESSED_DIR=/app/data/processed
OUTPUT_DIR=/app/data/outputs
MODEL_CHECKPOINT_PATH=/app/model/models/checkpoints/best_model.pt
```

### Monitoring
- Backend logs: `docker logs -f vidsum-backend`
- Database logs: `docker logs -f vidsum-postgres`
- Processing metrics: http://localhost:8000/api/videos

## 8. Troubleshooting

### Database Connection Errors
```bash
# Check PostgreSQL is running
psql -U postgres -d vidsum_gnn -c "SELECT version();"

# Verify DATABASE_URL in config.py matches your setup
# Common issues: wrong port, password, database name
```

### Model Loading Errors
```bash
# Verify checkpoint exists and is valid PyTorch file
python -c "import torch; print(torch.load('model/models/checkpoints/best_model.pt').keys())"

# Re-download pre-trained models if corrupted
rm -rf ~/.cache/huggingface/hub
# Then restart backend to re-download
```

### Out of Memory
```bash
# Reduce batch size or use CPU inference
# In model_service.py:
self.device = "cpu"  # Force CPU

# Or process fewer shots at once
top_k = 5  # Reduce from default 15
```

### Slow Processing
- **Shot Detection**: Takes ~1min per 10min video
- **Feature Extraction**: ~2min per 100 shots (GPU), ~10min (CPU)
- **GNN Inference**: ~5s per video
- **Text Summarization**: ~30s per video (depends on transcript length)

Optimize by:
- Using GPU for all models
- Reducing video resolution before upload
- Caching intermediate results

## 9. Development Workflow

### Adding New Summary Types
1. Update [vidsum_gnn/model_service.py](vidsum_gnn/model_service.py#L250) `LLMSummarizer.summarize()` to handle new type
2. Add radio button in [frontend/src/pages/DashboardPage.tsx](frontend/src/pages/DashboardPage.tsx#L280)
3. Update database schema if needed (migration)

### Changing GNN Architecture
1. Update [vidsum_gnn/model_service.py](vidsum_gnn/model_service.py#L30) `VidSumGNN` class
2. Re-train model with new architecture
3. Save checkpoint to `model/models/checkpoints/best_model.pt`
4. Restart backend

### Adding New Features
- Visual: Modify [vidsum_gnn/features/visual.py](vidsum_gnn/features/visual.py)
- Audio: Modify [vidsum_gnn/features/audio.py](vidsum_gnn/features/audio.py)
- Update feature dimension in `VidSumGNN(in_channels=NEW_DIM)`

## Support
- Documentation: [README.md](README.md)
- Training details: [model/train.ipynb](model/train.ipynb)
- API reference: http://localhost:8000/docs
