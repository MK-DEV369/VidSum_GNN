"""
Model Service - Loads trained GNN model and LLM summarizer for inference

⚠️ DEPRECATED: This module is deprecated and will be removed in a future version.
Please use the new inference module instead:

    from vidsum_gnn.inference.service import get_inference_service
    
    service = get_inference_service()
    scores, summary = service.process_video_pipeline(...)

Migration path:
- AudioTranscriber → vidsum_gnn.inference.transcription.WhisperTranscriber
- TextEmbedder → vidsum_gnn.inference.text_embedding.TextEmbedder
- LLMSummarizer → vidsum_gnn.inference.summarization.FlanT5Summarizer
- ModelService → vidsum_gnn.inference.service.InferenceService
- get_model_service() → get_inference_service()
"""
import warnings
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer
import json
import re

from vidsum_gnn.core.config import settings
from vidsum_gnn.utils.logging import get_logger
from vidsum_gnn.graph.model import VidSumGNN

logger = get_logger(__name__)


class AudioTranscriber:
    """Whisper ASR for speech-to-text"""
    def __init__(self, model_name="openai/whisper-base", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"✓ Whisper loaded on {self.device}")
    
    def transcribe(self, audio_path: Path, cache_dir: Optional[Path] = None) -> str:
        """Transcribe audio file to text"""
        cache_file = None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{audio_path.stem}_transcript.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)["text"]
        
        # Check if file exists and has content
        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return ""
        
        if audio_path.stat().st_size == 0:
            logger.warning(f"Audio file is empty: {audio_path}")
            return ""
        
        try:
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            
            # Check if audio has content
            if len(audio) == 0:
                logger.warning(f"Audio loaded but empty: {audio_path}")
                return ""
            
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(inputs["input_features"], max_new_tokens=128)
            
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if cache_file:
                with open(cache_file, "w") as f:
                    json.dump({"text": transcription}, f)
            
            return transcription
        except Exception as e:
            logger.error(f"Transcription error for {audio_path}: {e}")
            return ""


class TextEmbedder:
    """Sentence-Transformer for text embeddings"""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"✓ Text embedder loaded on {self.device}")
    
    def encode(self, text: str, cache_path: Optional[Path] = None) -> np.ndarray:
        """Encode text to embedding vector"""
        if cache_path and cache_path.exists():
            return np.load(cache_path)
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embedding)
        
        return embedding


class LLMSummarizer:
    """Flan-T5 for text summarization (free, local)"""
    def __init__(self, model_path: str = "google/flan-t5-base", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.max_input_chars = 1500
        logger.info(f"✓ LLM summarizer (Flan-T5) loaded on {self.device}")
    
    def summarize(
        self,
        transcripts: List[str],
        gnn_scores: List[float],
        summary_type: str = "balanced",
        text_length: str = "medium",
        summary_format: str = "bullet",
        top_k: int = 10
    ) -> str:
        """
        Generate text summary from top-scored shot transcripts.
        
        Args:
            transcripts: List of shot transcriptions
            gnn_scores: GNN importance scores for each shot
            summary_type: Type of summary (balanced, visual, audio, highlight)
            text_length: short (50-100 words), medium (100-200 words), long (200-400 words)
            summary_format: bullet, structured, or plain
            top_k: Number of top shots to include
        
        Returns:
            Single formatted summary string
        """
        # Set length parameters based on text_length
        length_map = {
            "short": (30, 70),
            "medium": (70, 150),
            "long": (150, 300)
        }
        min_length, max_length = length_map.get(text_length, (70, 150))
        # Normalize scores
        scores_arr = np.array(gnn_scores)
        if scores_arr.max() > scores_arr.min():
            scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
        else:
            scores_norm = scores_arr
        
        # Rank and select top-K shots
        indices = np.argsort(scores_norm)[::-1][:top_k]
        selected_transcripts = [transcripts[i] for i in indices if i < len(transcripts)]
        
        # Concatenate and truncate
        combined_text = " ".join(selected_transcripts)
        if len(combined_text) > self.max_input_chars:
            combined_text = combined_text[:self.max_input_chars]
        
        if not combined_text.strip():
            # No speech detected - generate visual summary from GNN scores
            fallback = {
                "bullet": "• Visual highlights selected by importance scoring\n• Key moments detected through shot analysis\n• No audio transcript available",
                "structured": "Summary: Video summary generated from visual shot importance analysis. Audio transcription not available.",
                "plain": "This is a visual summary generated from shot importance analysis. The most important shots have been selected based on visual features and their relationships in the video."
            }
            return fallback[summary_format]
        
        # Generate summary
        prompt = f"Summarize the following video content in {min_length}-{max_length} words:\n\n{combined_text}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                min_length=min_length,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Format according to requested format
        if summary_format == "bullet":
            return self._to_bullet_points(summary_text, text_length)
        elif summary_format == "structured":
            return self._to_structured(summary_text, summary_type, text_length)
        else:  # plain
            return summary_text
    
    def _to_bullet_points(self, text: str, text_length: str) -> str:
        """Convert summary to bullet points with varying detail"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Vary number of bullets by length
        num_bullets = {"short": 3, "medium": 5, "long": 8}.get(text_length, 5)
        return "\n".join([f"• {sent}" for sent in sentences[:num_bullets]])
    
    def _to_structured(self, text: str, summary_type: str, text_length: str) -> str:
        """Create structured summary with metadata and sections"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if text_length == "short":
            return f"**{summary_type.title()} Summary**\n\n{'. '.join(sentences[:2])}.\n\n*Generated by VIDSUM-GNN*"
        elif text_length == "long":
            intro = sentences[0] if sentences else text
            body = ". ".join(sentences[1:4]) if len(sentences) > 1 else ""
            conclusion = ". ".join(sentences[4:]) if len(sentences) > 4 else ""
            
            result = f"**{summary_type.title()} Summary**\n\n"
            result += f"**Overview:** {intro}.\n\n"
            if body:
                result += f"**Key Points:** {body}.\n\n"
            if conclusion:
                result += f"**Additional Details:** {conclusion}.\n\n"
            result += "*Generated by VIDSUM-GNN using Flan-T5*"
            return result
        else:  # medium
            return f"**{summary_type.title()} Summary**\n\n{text}\n\n*Generated by VIDSUM-GNN using Flan-T5*"


class ModelService:
    """
    Unified model service for video summarization inference
    Loads trained GNN, Whisper ASR, and Flan-T5 summarizer
    """
    def __init__(
        self,
        model_checkpoint: Optional[Path] = None,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load GNN model
        if model_checkpoint is None:
            model_checkpoint = Path(settings.MODEL_DIR) / "results" / "vidsumgnn_final.pt"
        
        self.gnn_model = self._load_gnn_model(model_checkpoint)
        
        # Load auxiliary models
        self.transcriber = AudioTranscriber(device=self.device)
        self.text_embedder = TextEmbedder(device=self.device)
        self.summarizer = LLMSummarizer(device=self.device)
        
        logger.info(f"✓ ModelService initialized on {self.device}")
    
    def _load_gnn_model(self, checkpoint_path: Path) -> VidSumGNN:
        """Load trained GNN model from checkpoint"""
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found at {checkpoint_path}, using untrained model")
            model = VidSumGNN(
                in_dim=1536,
                hidden_dim=512,
                num_heads=4,
                dropout=0.3
            )
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model = VidSumGNN(
                in_dim=1536,
                hidden_dim=512,
                num_heads=4,
                dropout=0.3
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✓ Loaded GNN from {checkpoint_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    @torch.no_grad()
    def predict_importance_scores(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> np.ndarray:
        """
        Run GNN inference to get importance scores
        
        Args:
            node_features: (N, 1536) shot features
            edge_index: (2, E) graph edges
        
        Returns:
            importance_scores: (N,) numpy array
        """
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        scores = self.gnn_model(node_features, edge_index)
        return scores.cpu().numpy().reshape(-1)
    
    def generate_text_summary(
        self,
        audio_paths: List[Path],
        gnn_scores: List[float],
        summary_type: str = "balanced",
        text_length: str = "medium",
        summary_format: str = "bullet",
        cache_dir: Optional[Path] = None
    ) -> str:
        """
        Generate textual summary from audio transcripts
        
        Args:
            audio_paths: Paths to shot audio files
            gnn_scores: GNN importance scores
            summary_type: Summary style
            text_length: short/medium/long
            summary_format: bullet/structured/plain
            cache_dir: Cache directory for transcripts
        
        Returns:
            Formatted summary string
        """
        # Transcribe audio
        transcripts = []
        for audio_path in audio_paths:
            audio_path_obj = Path(audio_path) if isinstance(audio_path, str) else audio_path
            if audio_path_obj.exists():
                try:
                    transcript = self.transcriber.transcribe(audio_path_obj, cache_dir)
                    transcripts.append(transcript)
                except Exception as e:
                    logger.error(f"Transcription failed for {audio_path_obj}: {e}")
                    transcripts.append("")
            else:
                transcripts.append("")
        
        # Generate summary
        summary = self.summarizer.summarize(
            transcripts,
            gnn_scores,
            summary_type=summary_type,
            text_length=text_length,
            summary_format=summary_format,
            top_k=10
        )
        
        return summary
    
    def process_video_end_to_end(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        audio_paths: List[Path],
        summary_type: str = "balanced",
        text_length: str = "medium",
        summary_format: str = "bullet",
        cache_dir: Optional[Path] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Complete pipeline: GNN scoring + text summarization
        
        Returns:
            (gnn_scores, text_summary)
        """
        # GNN inference
        gnn_scores = self.predict_importance_scores(node_features, edge_index)
        
        # Text summarization
        text_summary = self.generate_text_summary(
            audio_paths,
            gnn_scores.tolist(),
            summary_type,
            text_length,
            summary_format,
            cache_dir
        )
        
        return gnn_scores, text_summary


# Global singleton
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """
    Get or create global model service instance
    
    ⚠️ DEPRECATED: Use get_inference_service() instead:
    
        from vidsum_gnn.inference.service import get_inference_service
        service = get_inference_service()
    """
    warnings.warn(
        "model_service.get_model_service() is deprecated. "
        "Use vidsum_gnn.inference.service.get_inference_service() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
