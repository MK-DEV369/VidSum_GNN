# Code Analysis & Refactoring Plan

## Current Issues Identified

### 1. **Duplicate Audio Processing**
- **Location**: `vidsum_gnn/processing/audio.py` vs `vidsum_gnn/features/audio.py`
- **Issue**: Two separate files handling audio
  - `processing/audio.py`: FFmpeg extraction only (utilities)
  - `features/audio.py`: HuBERT feature encoding
- **Resolution**: ✅ **CORRECT SEPARATION** - These serve different purposes:
  - `processing/` = Data pipeline (audio extraction)
  - `features/` = ML feature extraction (HuBERT embeddings)

### 2. **Model Service vs Graph Model Confusion**
- **Files**: `vidsum_gnn/model_service.py` vs `vidsum_gnn/graph/model.py`
- **Current State**:
  - `graph/model.py`: Contains **VidSumGNN** (GAT model architecture)
  - `model_service.py`: Contains:
    - AudioTranscriber (Whisper)
    - TextEmbedder (Sentence-Transformers)
    - LLMSummarizer (Flan-T5)
    - ModelService (orchestrator)
- **Issue**: `model_service.py` duplicates functionality already in dedicated modules
- **Resolution**: **REFACTOR NEEDED**

### 3. **Training Notebook vs Production Code Mismatch**
- **Issue**: `model/train.ipynb` has complete training pipeline but not integrated into backend
- **Training Components in Notebook**:
  - VideoDatasetLoader
  - FeatureExtractor (ViT + HuBERT)
  - VidSumGNN (identical to graph/model.py)
  - Training loop with validation
  - AudioTranscriber (Whisper)
  - TextEmbedder (Sentence-Transformers)
  - SemanticFusion (NOT in backend)

---

## Optimal Directory Structure

```
vidsum_gnn/
├── api/                          # FastAPI routes & tasks
│   ├── main.py                   # App initialization, WebSocket manager
│   ├── routes.py                 # Upload, download, status endpoints
│   └── tasks.py                  # Background processing orchestration
│
├── core/                         # Core configuration
│   ├── config.py                 # Settings (paths, model configs)
│   └── __init__.py
│
├── db/                           # Database layer
│   ├── client.py                 # SQLAlchemy session management
│   ├── models.py                 # Video, Shot, Summary ORM models
│   └── __init__.py
│
├── processing/                   # Video/Audio preprocessing pipeline
│   ├── video.py                  # FFmpeg: probe, transcode
│   ├── audio.py                  # FFmpeg: extract audio segments
│   └── shot_detection.py        # PySceneDetect: shot boundaries
│
├── features/                     # ML Feature Extractors
│   ├── visual.py                 # ViT-B/16 (768-dim)
│   ├── audio.py                  # HuBERT (768-dim)
│   └── __init__.py
│
├── graph/                        # Graph Neural Network
│   ├── model.py                  # VidSumGNN (GAT architecture)
│   ├── builder.py                # Build PyG Data from shots
│   └── __init__.py
│
├── inference/                    # 🆕 NEW: Unified Inference Service
│   ├── model_manager.py          # Model loading & caching
│   ├── transcription.py          # Whisper ASR
│   ├── text_embedding.py         # Sentence-Transformers
│   ├── summarization.py          # Flan-T5 text generation
│   └── service.py                # End-to-end inference pipeline
│
├── summary/                      # Summary generation
│   ├── assembler.py              # FFmpeg video assembly
│   └── __init__.py
│
├── training/                     # 🆕 NEW: Training utilities
│   ├── dataset.py                # VideoDatasetLoader from notebook
│   ├── trainer.py                # Training loop from notebook
│   └── __init__.py
│
└── utils/                        # Utilities
    ├── logging.py                # Structured logging
    └── __init__.py
```

---

## Refactoring Actions

### ✅ Phase 1: Extract Inference Components (PRIORITY)

#### 1.1 Create `inference/transcription.py`
**Extract from**: `model_service.py` (AudioTranscriber class)
**Purpose**: Whisper ASR for audio-to-text
```python
# vidsum_gnn/inference/transcription.py
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import librosa

class WhisperTranscriber:
    def __init__(self, model_name="openai/whisper-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def transcribe(self, audio_path: Path, cache_dir: Path = None) -> str:
        # Implementation from model_service.py
        ...
```

#### 1.2 Create `inference/text_embedding.py`
**Extract from**: `model_service.py` (TextEmbedder class)
```python
# vidsum_gnn/inference/text_embedding.py
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        ...
```

#### 1.3 Create `inference/summarization.py`
**Extract from**: `model_service.py` (LLMSummarizer class)
```python
# vidsum_gnn/inference/summarization.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class FlanT5Summarizer:
    def __init__(self, model_path="google/flan-t5-base", device=None):
        ...
```

#### 1.4 Create `inference/model_manager.py`
**Purpose**: Lazy loading & caching of all models
```python
# vidsum_gnn/inference/model_manager.py
from typing import Optional
import torch

class ModelManager:
    """Singleton manager for all inference models"""
    _instance = None
    
    def __init__(self):
        self._gnn_model = None
        self._whisper = None
        self._text_embedder = None
        self._summarizer = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_gnn_model(self, checkpoint_path=None):
        if self._gnn_model is None:
            from vidsum_gnn.graph.model import VidSumGNN
            # Load model...
        return self._gnn_model
    
    # Similar for other models...
```

#### 1.5 Create `inference/service.py`
**Purpose**: Main inference orchestration
```python
# vidsum_gnn/inference/service.py
from pathlib import Path
import torch
import numpy as np
from typing import List, Tuple

from vidsum_gnn.inference.model_manager import ModelManager
from vidsum_gnn.inference.transcription import WhisperTranscriber
from vidsum_gnn.inference.summarization import FlanT5Summarizer

class InferenceService:
    """
    End-to-end inference pipeline:
    1. GNN importance scoring
    2. Audio transcription
    3. Text summarization
    """
    def __init__(self):
        self.manager = ModelManager.get_instance()
    
    def predict_importance_scores(
        self, 
        node_features: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> np.ndarray:
        """Run GNN inference"""
        model = self.manager.get_gnn_model()
        with torch.no_grad():
            scores = model(node_features, edge_index)
        return scores.cpu().numpy()
    
    def generate_summary(
        self,
        audio_paths: List[Path],
        gnn_scores: List[float],
        summary_type: str = "balanced",
        text_length: str = "medium",
        summary_format: str = "bullet"
    ) -> str:
        """Generate text summary from audio + GNN scores"""
        # 1. Transcribe
        transcriber = self.manager.get_whisper()
        transcripts = [transcriber.transcribe(p) for p in audio_paths]
        
        # 2. Summarize
        summarizer = self.manager.get_summarizer()
        summary = summarizer.summarize(
            transcripts, gnn_scores, 
            summary_type, text_length, summary_format
        )
        return summary
    
    def process_video_pipeline(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        audio_paths: List[Path],
        **config
    ) -> Tuple[np.ndarray, str]:
        """Complete pipeline"""
        scores = self.predict_importance_scores(node_features, edge_index)
        summary = self.generate_summary(audio_paths, scores.tolist(), **config)
        return scores, summary


# Global accessor
_service = None
def get_inference_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service
```

---

### ✅ Phase 2: Extract Training Code

#### 2.1 Create `training/dataset.py`
**Extract from**: `model/train.ipynb` Cell 2 (VideoDatasetLoader)
```python
# vidsum_gnn/training/dataset.py
from pathlib import Path
import pandas as pd

class VideoDatasetLoader:
    """Universal loader for TVSum, SumMe, YouTube datasets"""
    def __init__(self, base_path='data/raw'):
        self.base_path = Path(base_path)
        self.videos = []
    # ... rest from notebook
```

#### 2.2 Create `training/trainer.py`
**Extract from**: `model/train.ipynb` Cells 13-14 (training loop)
```python
# vidsum_gnn/training/trainer.py
import torch
from torch.utils.data import DataLoader

class GNNTrainer:
    """Training pipeline for VidSumGNN"""
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
    
    def train_epoch(self, dataloader, scaler=None):
        # From notebook...
        pass
    
    def validate(self, dataloader):
        pass
    
    def train(self, train_loader, val_loader, epochs, checkpoint_dir):
        # Full training loop from notebook
        pass
```

---

### ✅ Phase 3: Update API to Use New Structure

#### 3.1 Update `api/tasks.py`
```python
# OLD
from vidsum_gnn.model_service import get_model_service

# NEW
from vidsum_gnn.inference.service import get_inference_service

async def process_video_task(video_id: str, config: dict):
    # ... preprocessing ...
    
    # Feature extraction (unchanged)
    vis_encoder = VisualEncoder()
    aud_encoder = AudioEncoder()
    vis_feats = vis_encoder.encode(keyframe_paths)
    aud_feats = aud_encoder.encode(audio_paths)
    features = torch.cat([vis_feats, aud_feats], dim=1)
    
    # Graph building (unchanged)
    builder = GraphBuilder()
    graph_data = builder.build_graph(shots_data, features)
    
    # NEW: Use InferenceService
    inference_service = get_inference_service()
    gnn_scores, text_summary = inference_service.process_video_pipeline(
        node_features=graph_data.x,
        edge_index=graph_data.edge_index,
        audio_paths=audio_paths,
        summary_type=config["summary_type"],
        text_length=config["text_length"],
        summary_format=config["summary_format"]
    )
    
    # ... rest unchanged ...
```

---

## Migration Checklist

### Immediate Actions
- [x] ✅ Analyze current codebase structure
- [ ] ⚠️ Create `vidsum_gnn/inference/` directory
- [ ] ⚠️ Move AudioTranscriber → `inference/transcription.py`
- [ ] ⚠️ Move TextEmbedder → `inference/text_embedding.py`
- [ ] ⚠️ Move LLMSummarizer → `inference/summarization.py`
- [ ] ⚠️ Create `inference/model_manager.py` (singleton)
- [ ] ⚠️ Create `inference/service.py` (main pipeline)
- [ ] ⚠️ Update `api/tasks.py` to use InferenceService
- [ ] ⚠️ DEPRECATE `model_service.py` (keep for backward compat initially)

### Training Integration
- [ ] ⚠️ Create `vidsum_gnn/training/` directory
- [ ] ⚠️ Extract VideoDatasetLoader → `training/dataset.py`
- [ ] ⚠️ Extract training loop → `training/trainer.py`
- [ ] ⚠️ Create training CLI script: `scripts/train_model.py`

### Testing & Validation
- [ ] ⚠️ Test GNN inference with new service
- [ ] ⚠️ Test Whisper transcription
- [ ] ⚠️ Test Flan-T5 summarization
- [ ] ⚠️ End-to-end pipeline test
- [ ] ⚠️ Load testing with concurrent requests

---

## Benefits of Refactoring

### 1. **Clear Separation of Concerns**
- `processing/` = Raw data extraction
- `features/` = ML feature extraction
- `inference/` = Trained model inference
- `training/` = Model training utilities

### 2. **Lazy Model Loading**
ModelManager ensures models are loaded only when needed and cached

### 3. **Testability**
Each component can be unit tested independently

### 4. **Scalability**
Easy to add new models (e.g., better ASR, different LLM)

### 5. **Maintainability**
Training code is now in codebase, not just notebook

---

## File Deletion Plan

### Safe to Delete (After Refactoring)
- `model_service.py` → Replace with `inference/service.py`

### Keep (Active Use)
- `processing/audio.py` ✅ (FFmpeg utilities)
- `features/audio.py` ✅ (HuBERT encoder)
- `graph/model.py` ✅ (GNN architecture)

---

## Summary

**Current State**: 
- Monolithic `model_service.py` mixing concerns
- Training code trapped in notebook
- Unclear boundaries between processing/features/inference

**Target State**:
- Clean separation: `processing` → `features` → `inference` → `summary`
- Training utilities in codebase
- ModelManager for efficient resource management
- InferenceService as single entry point

**Next Step**: Execute Phase 1 (Create inference/ modules)
