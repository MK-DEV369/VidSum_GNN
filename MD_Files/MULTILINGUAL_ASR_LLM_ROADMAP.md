# 🌍 Multilingual ASR + Text Summarization Roadmap

## Executive Summary

Current pipeline produces **extractive video summaries** (shot selection + MP4 output). To add **multilingual text summaries with user-controlled length**, integrate:

1. **ASR (Automatic Speech Recognition)**: Whisper (multilingual) → per-shot transcripts
2. **Translation (Optional)**: Translate transcripts to pivot language (English) for consistency
3. **Text Embeddings**: Sentence-Transformers → semantic features
4. **LLM Summarization**: Claude/GPT with explicit length constraints (e.g., 5–8 bullet points)
5. **GNN Enhancement (Optional)**: Enrich graph nodes with text embeddings for better multimodal fusion

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Existing Pipeline (Keep)                     │
│  Video → Shot Detection → Visual/Audio Features → GNN Scoring   │
│  Output: Extractive video summary (MP4)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  🆕 NEW: Text Extraction Layer  │
        └────────────────┬────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌─────────┐      ┌──────────────┐      ┌──────────────┐
│  ASR    │      │  Translation │      │ Text Features│
│(Whisper)│      │   (Optional) │      │(Transformers)│
└────┬────┘      └──────┬───────┘      └──────┬───────┘
     │                  │                     │
     │  Per-Shot        │                     │
     │  Transcripts     ▼                     │
     │           Normalized text              │
     │                                        │
     └────────────────┬───────────────────────┘
                      │
              ┌───────▼────────┐
              │  Text Features │
              │  (768-dim or   │
              │   1536-dim)    │
              └───────┬────────┘
                      │
         ┌────────────▼────────────┐
         │ Concatenate with Visual │
         │ (Optional: 2304-dim)    │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │ GNN Re-training         │
         │ (Multimodal nodes)      │
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │ Enhanced Shot Scores    │
         └────────────┬────────────┘
                      │
    ┌─────────────────┼──────────────────┐
    │                 │                  │
    ▼                 ▼                  ▼
┌─────────┐  ┌──────────────┐  ┌────────────────┐
│ Selected│  │   Selected   │  │  LLM Prompt    │
│ Shots   │  │ Transcripts  │  │ with Length    │
│ (GNN)   │  │ (Time-order) │  │ Constraints    │
└────┬────┘  └──────┬───────┘  └────────┬───────┘
     │              │                    │
     └──────────────┼────────────────────┘
                    │
         ┌──────────▼─────────┐
         │  LLM (Claude/GPT)  │
         │ "Generate 5-8      │
         │  bullet summary"   │
         └──────────┬─────────┘
                    │
    ┌───────────────┼────────────────┐
    │               │                │
    ▼               ▼                ▼
┌──────────┐  ┌──────────┐  ┌───────────────┐
│ Headline │  │ Bullets  │  │ Full Text     │
│(1-2 line)│  │(5-8 line)│  │ (Optional)    │
└──────────┘  └──────────┘  └───────────────┘
```

---

## Phase 1: ASR Implementation (Whisper)

### 1.1 Setup

**Install Whisper**:
```bash
pip install openai-whisper torch torchvision torchaudio
```

**Supported languages**: 99+ languages (Arabic, Mandarin, French, Spanish, Hindi, etc.)

### 1.2 Create ASR Module

**File**: `vidsum_gnn/features/asr.py`

```python
import whisper
import torch
from pathlib import Path
from typing import Dict, List, Optional

class AudioTranscriber:
    def __init__(self, model_size: str = "base", device: str = "cuda"):
        """
        model_size: 'tiny', 'base', 'small', 'medium', 'large'
        Tradeoff: speed vs accuracy. 'base' is good default.
        """
        self.model = whisper.load_model(model_size, device=device)
        self.device = device
    
    def transcribe_shot(
        self, 
        audio_path: Path, 
        start_sec: float, 
        end_sec: float, 
        language: Optional[str] = None  # e.g., 'en', 'fr', 'zh'
    ) -> Dict:
        """
        Transcribe a single shot's audio.
        Returns: {
            'text': str,
            'language': str,
            'confidence': float (0-1),
            'segments': List[{start, end, text}]
        }
        """
        # Load audio segment
        import librosa
        y, sr = librosa.load(str(audio_path), sr=16000, 
                            offset=start_sec, duration=end_sec - start_sec)
        
        # Transcribe
        result = self.model.transcribe(
            y,
            language=language,
            fp16=torch.cuda.is_available()
        )
        
        return {
            'text': result['text'],
            'language': result['language'],
            'confidence': sum(seg['confidence'] for seg in result['segments']) / len(result['segments']) if result['segments'] else 0.9,
            'segments': result['segments']
        }
    
    def transcribe_video(
        self, 
        audio_path: Path, 
        shots: List[Dict],
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        Transcribe all shots in a video.
        shots: [{start_sec, end_sec, ...}, ...]
        Returns: List of transcript dicts (one per shot)
        """
        transcripts = []
        for i, shot in enumerate(shots):
            print(f"Transcribing shot {i+1}/{len(shots)}...")
            transcript = self.transcribe_shot(
                audio_path, 
                shot['start_sec'], 
                shot['end_sec'],
                language=language
            )
            transcripts.append(transcript)
        return transcripts
```

### 1.3 Integration Points

**In `vidsum_gnn/api/tasks.py` (during shot processing)**:

```python
from vidsum_gnn.features.asr import AudioTranscriber

async def process_video_task(video_id: str, config: dict):
    # ... existing code (shot detection, visual/audio features) ...
    
    # NEW: Transcribe shots
    transcriber = AudioTranscriber(model_size="base")
    transcripts = transcriber.transcribe_video(
        audio_path=audio_path,
        shots=shots,
        language=config.get('language', None)  # User-specified or auto-detect
    )
    
    # Store transcripts with shots
    for shot, transcript in zip(shots, transcripts):
        shot['transcript'] = transcript['text']
        shot['language'] = transcript['language']
        shot['transcript_confidence'] = transcript['confidence']
    
    # ... rest of pipeline (GNN scoring, selection, assembly) ...
```

### 1.4 Performance Notes

| Model | Speed | Accuracy | Latency (per min of audio) |
|-------|-------|----------|---------------------------|
| tiny | ⚡⚡⚡ | 60% | ~6 sec |
| base | ⚡⚡ | 75% | ~15 sec |
| small | ⚡ | 85% | ~30 sec |
| medium | - | 92% | ~60 sec |
| large | 🐢 | 96% | ~90 sec |

**Recommendation**: Use `base` for real-time; `small` for high accuracy.

---

## Phase 2: Translation (Optional)

### 2.1 Setup

**Install translation library**:
```bash
pip install transformers torch
```

### 2.2 Create Translation Module

**File**: `vidsum_gnn/features/translation.py`

```python
from transformers import MarianMTModel, MarianTokenizer
from typing import List, Optional

class Translator:
    def __init__(self, src_lang: str, tgt_lang: str = "en"):
        """
        Translate from src_lang to tgt_lang.
        src_lang: 'es', 'fr', 'de', 'zh', 'ar', etc.
        tgt_lang: target (default 'en' for English pivot)
        
        Model format: 'Helsinki-NLP/Opus-MT-{src}-{tgt}'
        """
        model_name = f"Helsinki-NLP/Opus-MT-{src_lang}-{tgt_lang}"
        self.model = MarianMTModel.from_pretrained(model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    def translate(self, texts: List[str]) -> List[str]:
        """Batch translate texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        translated = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(translated, skip_special_tokens=True)
```

### 2.3 Usage

```python
# Detect language from Whisper output, then translate
if transcript['language'] != 'en':
    translator = Translator(
        src_lang=transcript['language'],
        tgt_lang='en'
    )
    transcript['text_en'] = translator.translate([transcript['text']])[0]
    transcript['original_language'] = transcript['language']
else:
    transcript['text_en'] = transcript['text']
    transcript['original_language'] = 'en'
```

---

## Phase 3: Text Embeddings

### 3.1 Setup

```bash
pip install sentence-transformers
```

### 3.2 Create Embedding Module

**File**: `vidsum_gnn/features/text_embedding.py`

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class TextEmbedder:
    def __init__(self, model: str = "multilingual-MiniLM-L6-v2"):
        """
        Multilingual sentence embedder (supports 100+ languages).
        Options:
        - 'multilingual-MiniLM-L6-v2' (384-dim, fast)
        - 'paraphrase-multilingual-mpnet-base-v2' (768-dim, better quality)
        """
        self.model = SentenceTransformer(model)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed list of texts.
        Returns: (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings
```

### 3.3 Usage

```python
embedder = TextEmbedder("multilingual-MiniLM-L6-v2")
text_embeddings = embedder.embed_texts(
    [shot['transcript']['text'] for shot in shots]
)
# text_embeddings shape: (num_shots, 384)
```

---

## Phase 4: Enhanced GNN Training

### 4.1 Multimodal Node Features

**Current**: 1536-dim (1024 visual + 512 audio? or actual concat)
**New**: Option A (concatenate) or Option B (separate branches)

#### Option A: Concatenate (Simpler)

```python
# Per shot:
visual_feat = (768,)           # ViT-B/16
audio_feat = (768,)            # Wav2Vec2
text_feat = (384,)             # Multilingual embeddings
─────────────────────────────
combined = (1920,)  # Concatenate all

# In GNN model definition:
GNN(in_dim=1920, hidden_dim=1024, ...)
```

#### Option B: Separate Branches (Flexible)

```python
class MultimodalGNN(nn.Module):
    def __init__(self, in_dim_v=768, in_dim_a=768, in_dim_t=384, hidden_dim=1024):
        # Three separate projection branches
        self.visual_proj = nn.Linear(in_dim_v, hidden_dim)
        self.audio_proj = nn.Linear(in_dim_a, hidden_dim)
        self.text_proj = nn.Linear(in_dim_t, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # GAT layers (unchanged)
        self.gat = GATv2Conv(hidden_dim, hidden_dim, ...)
    
    def forward(self, x_v, x_a, x_t, edge_index, edge_attr):
        # Project each modality
        h_v = self.visual_proj(x_v)
        h_a = self.audio_proj(x_a)
        h_t = self.text_proj(x_t)
        
        # Fuse
        h = self.fusion(torch.cat([h_v, h_a, h_t], dim=1))
        
        # GAT layers
        h = self.gat(h, edge_index, edge_attr)
        return h
```

### 4.2 Training Updates

**In `train.ipynb`**:

```python
# Load transcripts and embeddings
transcripts = load_transcripts(dataset)  # Per-shot texts
text_embeddings = embed_texts(transcripts)  # (N_shots, 384)

# Build enhanced dataset
for video in dataset:
    for i, shot in enumerate(video['shots']):
        shot['text_embedding'] = text_embeddings[shot_idx]

# Train with multimodal node features
model = MultimodalGNN(in_dim_v=768, in_dim_a=768, in_dim_t=384)
# ... training loop with 3-modality input ...
```

---

## Phase 5: User Preference Conditioning

### 5.1 User Config

```python
class SummaryRequest(BaseModel):
    video_id: str
    target_duration_sec: int = 60
    text_summary_length: str = "medium"  # "short" (3-4), "medium" (5-7), "long" (8-10) lines
    style: str = "informative"  # "highlight" or "informative"
    modality_bias: float = 0.5  # 0=favor visual, 1=favor speech
    language: Optional[str] = None  # 'en', 'fr', 'zh', etc. for ASR + output
    output_format: str = "video+text"  # "video", "text", or "video+text"
```

### 5.2 Score Adjustment

**In selection step**:

```python
def adjust_shot_scores(
    gnn_scores: np.ndarray,
    speech_density: np.ndarray,  # Computed from transcripts
    keyword_hits: np.ndarray,    # TF-IDF or BM25 match to user query
    modality_bias: float = 0.5,
) -> np.ndarray:
    """
    Reweight GNN scores based on user preferences.
    score' = α*gnn + β*speech_density + γ*keyword_match
    where β, γ depend on modality_bias and user config.
    """
    alpha = 0.7
    beta = modality_bias * 0.2
    gamma = 0.1
    
    adjusted = (
        alpha * gnn_scores +
        beta * speech_density +
        gamma * keyword_hits
    )
    return adjusted / adjusted.sum()  # Normalize
```

---

## Phase 6: LLM-Based Text Summarization

### 6.1 Setup

```bash
pip install anthropic
# or
pip install openai
```

### 6.2 Create LLM Summarizer

**File**: `vidsum_gnn/summary/text_summarizer.py`

```python
from typing import List, Dict, Optional
import anthropic  # or openai

class TextSummarizer:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def generate_summary(
        self,
        transcripts: List[str],
        shot_scores: List[float],
        summary_length: str = "medium",
        language: str = "en"
    ) -> Dict:
        """
        Generate text summary from ordered transcripts.
        
        summary_length: "short" (3-4), "medium" (5-7), "long" (8-10)
        """
        
        # Build context with weighted importance
        context = "\n\n".join([
            f"[Importance: {score:.2f}]\n{text}"
            for text, score in zip(transcripts, shot_scores)
        ])
        
        # Map length to bullet count
        length_map = {
            "short": (3, 4),
            "medium": (5, 7),
            "long": (8, 10)
        }
        min_bullets, max_bullets = length_map.get(summary_length, (5, 7))
        
        prompt = f"""You are a video summarization expert. 
        
Below is a transcript from a video (with importance scores). 
Generate a summary as a NUMBERED LIST with {min_bullets}-{max_bullets} bullet points.

Focus on the most important points (indicated by higher scores).
Keep each point concise (1-2 sentences max).
Output ONLY the numbered list, no preamble.

TRANSCRIPT WITH IMPORTANCE SCORES:
{context}

SUMMARY (in {language}):
"""
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        summary_text = message.content[0].text
        
        return {
            "text": summary_text,
            "model": self.model,
            "length": summary_length,
            "bullet_count_requested": f"{min_bullets}-{max_bullets}",
            "language": language
        }
    
    def generate_headline(self, transcripts: List[str]) -> str:
        """Generate 1-2 line headline."""
        context = " ".join(transcripts)
        
        prompt = f"""Summarize this video in 1-2 sentences (max 20 words each):
{context}

HEADLINE:"""
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text.strip()
```

### 6.3 Integration

**In `vidsum_gnn/api/routes.py`**:

```python
@router.post("/summarize/{video_id}")
async def get_text_summary(
    video_id: str,
    request: SummaryRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate text summary for a video."""
    
    # Get video and selected shots
    summary = await db.get(Summary, f"{video_id}_summary")
    shots = await db.execute(
        select(Shot).where(Shot.summary_id == summary.summary_id)
    )
    
    # Extract transcripts in order
    transcripts = [shot.transcript for shot in sorted(shots, key=lambda s: s.start_sec)]
    shot_scores = [shot.importance_score for shot in sorted(shots, key=lambda s: s.start_sec)]
    
    # Generate summaries
    summarizer = TextSummarizer()
    
    headline = summarizer.generate_headline(transcripts)
    text_summary = summarizer.generate_summary(
        transcripts,
        shot_scores,
        summary_length=request.text_summary_length,
        language=request.language or "en"
    )
    
    return {
        "video_id": video_id,
        "headline": headline,
        "summary": text_summary,
        "format": "text"
    }
```

---

## Phase 7: API Endpoint Updates

### 7.1 Enhanced Upload Endpoint

```python
@router.post("/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    target_duration: int = 60,
    text_summary_length: str = "medium",
    style: str = "informative",
    language: Optional[str] = None,
    modality_bias: float = 0.5,
    output_format: str = "video+text",  # "video", "text", or "video+text"
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db)
):
    """Upload video and request summarization."""
    
    video_id = str(uuid.uuid4())
    
    # Store request config
    config = {
        "target_duration": target_duration,
        "text_summary_length": text_summary_length,
        "style": style,
        "language": language,
        "modality_bias": modality_bias,
        "output_format": output_format,
    }
    
    # ... file upload logic ...
    
    # Start background processing (with new parameters)
    background_tasks.add_task(
        process_video_with_text,
        video_id,
        config
    )
    
    return {"video_id": video_id, "status": "queued"}
```

### 7.2 Results Endpoint

```python
@router.get("/results/{video_id}")
async def get_results(video_id: str, db: AsyncSession = Depends(get_db)):
    """Get both video and text summaries."""
    
    summary = await db.get(Summary, f"{video_id}_summary")
    
    return {
        "video_id": video_id,
        "video_summary": {
            "path": summary.video_path,
            "duration_sec": summary.duration_sec,
            "num_shots": summary.num_shots
        },
        "text_summary": {
            "headline": summary.headline,
            "bullets": summary.text_summary,
            "length": summary.text_length_type,
            "language": summary.output_language
        },
        "metadata": {
            "total_duration": summary.original_duration,
            "compression_ratio": summary.compression_ratio
        }
    }
```

---

## Implementation Timeline

| Phase | Task | Effort | Duration |
|-------|------|--------|----------|
| 1 | ASR (Whisper) integration | Medium | 2-3 days |
| 2 | Translation (optional) | Low | 1 day |
| 3 | Text embeddings | Low | 1 day |
| 4 | GNN multimodal training | High | 3-5 days |
| 5 | User preference conditioning | Medium | 2 days |
| 6 | LLM text summarization | Medium | 2-3 days |
| 7 | API updates | Medium | 2 days |
| **Total** | | | **13-18 days** |

---

## Database Schema Updates

### New Tables

```sql
-- Transcripts
CREATE TABLE transcripts (
    id SERIAL PRIMARY KEY,
    shot_id BIGINT NOT NULL REFERENCES shots(id),
    text TEXT NOT NULL,
    language VARCHAR(10),
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Text Summaries
CREATE TABLE text_summaries (
    id SERIAL PRIMARY KEY,
    summary_id BIGINT NOT NULL REFERENCES summaries(id),
    headline TEXT,
    bullets TEXT,  -- JSON list or plain text
    full_text TEXT,
    length_type VARCHAR(20),  -- "short", "medium", "long"
    output_language VARCHAR(10),
    model_used VARCHAR(100),  -- "claude-3-sonnet", etc.
    created_at TIMESTAMP DEFAULT NOW()
);

-- Text Embeddings
CREATE TABLE text_embeddings (
    id SERIAL PRIMARY KEY,
    shot_id BIGINT NOT NULL REFERENCES shots(id),
    embedding VECTOR(384),  -- pgvector type
    model_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Best Practices

### 1. ASR Quality
- Test on sample videos first (small budget)
- Use language auto-detection (Whisper is reliable)
- Cache transcripts (expensive operation)
- Fallback: if ASR fails, skip text summary

### 2. LLM Costs
- Use shorter models for speed (Claude Sonnet, GPT-3.5 Turbo)
- Batch summaries (process 10 videos at once)
- Estimate: ~5-10 cents per video for LLM summarization

### 3. Caching Strategy
```python
# Cache transcripts and embeddings
transcript_cache = f"data/cache/{video_id}_transcripts.json"
if Path(transcript_cache).exists():
    transcripts = json.load(open(transcript_cache))
else:
    transcripts = transcriber.transcribe_video(...)
    json.dump(transcripts, open(transcript_cache, 'w'))
```

### 4. Multilingual Support
- Whitelist supported languages (Whisper: ~99)
- Translate to English for LLM if needed (better quality)
- Output text in user's requested language

### 5. Error Handling
```python
try:
    transcripts = transcriber.transcribe_video(audio_path, shots)
except Exception as e:
    logger.warning(f"ASR failed: {e}")
    transcripts = [{"text": "[No speech]", "language": "unknown"} for _ in shots]
    # Continue without text summary
```

---

## Testing Checklist

- [ ] ASR on multilingual videos (English, Spanish, French, Mandarin)
- [ ] Translation accuracy (translate to English and back)
- [ ] Text embedding quality (similar shots have similar embeddings)
- [ ] GNN training with multimodal features (convergence check)
- [ ] LLM summarization length control (actually generates 5-7 bullets)
- [ ] API endpoint with text summary output
- [ ] End-to-end test: upload video → receive video + text summary
- [ ] Cost estimation: run on 10 videos, measure API spend
- [ ] Latency: measure total time for full pipeline

---

## Future Enhancements

1. **Keyword Extraction**: Extract key terms per shot for better targeting
2. **Abstractive Video Summarization**: Use extractive shots as input to video synthesis (create new summary video with AI narration)
3. **Interactive Summaries**: Let users click on bullets to jump to corresponding video sections
4. **Multi-language Output**: Generate summaries in multiple languages simultaneously
5. **Speaker Identification**: Diarize and identify speakers in ASR output
6. **Sentiment Analysis**: Tag shots by sentiment (positive, negative, neutral)
7. **Entity Recognition**: Extract entities (people, places, objects) from transcripts

---

## File Structure (After Implementation)

```
vidsum_gnn/
├── features/
│   ├── visual.py
│   ├── audio.py
│   ├── asr.py                    # NEW
│   ├── translation.py             # NEW (optional)
│   └── text_embedding.py          # NEW
├── summary/
│   ├── selector.py
│   ├── assembler.py
│   └── text_summarizer.py         # NEW
├── graph/
│   ├── model.py                   # UPDATE (multimodal)
│   └── builder.py
├── api/
│   ├── main.py
│   ├── routes.py                  # UPDATE
│   └── tasks.py                   # UPDATE
└── db/
    ├── models.py                  # UPDATE (new tables)
    └── init_timescaledb.sql       # UPDATE
```

---

## Summary

Your current model is **not wasted**. The VidSumGNN extracts important shots beautifully. By layering ASR + LLM on top, you get:

✅ **Extractive video summary** (existing: MP4 video output)
✅ **Text summary** with user-controlled length (NEW: 3-10 bullet points)
✅ **Multilingual support** (Whisper + translation)
✅ **User preferences** (style, modality bias, language)
✅ **Multimodal GNN** for potential performance gains (optional upgrade)

**Timeline**: 13–18 days to full integration
**MVP (Minimal Viable Product)**: 5–7 days (ASR + LLM only, skip multimodal GNN)
