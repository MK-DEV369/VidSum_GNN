"""
Main Inference Service - End-to-end video summarization pipeline.
Orchestrates GNN scoring, transcription, and text summarization.

SUPPORTS:
- Summary Types: balanced, visual_priority, audio_priority, highlights
- Text Lengths: short, medium, long
- Formats: bullet, structured, plain
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from vidsum_gnn.inference.model_manager import ModelManager
from vidsum_gnn.utils.logging import get_logger
from vidsum_gnn.core.config import settings

logger = get_logger(__name__)


class InferenceService:
    """
    End-to-end inference pipeline for video summarization:
    1. GNN importance scoring (binary classification per shot)
    2. Audio transcription (Whisper ASR)
    3. Text summarization (Flan-T5 with context-aware prompting)
    4. Multiple format outputs (bullet, structured, plain)
    5. Multiple content types (balanced, visual, audio, highlights)
    """
    
    def __init__(self):
        """Initialize inference service with model manager"""
        self.manager = ModelManager.get_instance()
        self._last_hidden: Optional[torch.Tensor] = None
        logger.info("InferenceService initialized")
    
    @torch.no_grad()
    def predict_importance_scores(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> np.ndarray:
        """
        Run GNN inference to get shot importance scores.
        
        Args:
            node_features: (N, 1536) shot features [ViT + HuBERT]
            edge_index: (2, E) graph edges
            
        Returns:
            importance_scores: (N,) numpy array
        """
        device = self.manager.get_device()
        # Ensure model matches feature dimension and load configured checkpoint
        model = self.manager.get_gnn_model(
            checkpoint_path=Path(settings.GNN_CHECKPOINT) if settings.GNN_CHECKPOINT else None,
            in_dim=int(node_features.shape[1]) if node_features is not None else None
        )
        
        # Move to device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        
        # Inference
        scores, h = model(node_features, edge_index)
        # Cache hidden for fusion
        try:
            self._last_hidden = h.detach().to('cpu')
        except Exception:
            self._last_hidden = None
        # Convert logits to probabilities for binary classification
        probs = torch.sigmoid(scores)
        
        # Return as numpy array
        return probs.cpu().numpy().reshape(-1)

    def fuse_hidden_with_text(self, transcripts: List[str]) -> Optional[torch.Tensor]:
        """
        Fuse cached hidden representation with text embeddings.
        Returns fused features tensor of shape (N, hidden_dim + 384) on CPU.
        """
        if self._last_hidden is None or not transcripts:
            return None
        try:
            embedder = self.manager.get_text_embedder()
            text_embs = embedder.batch_encode(transcripts)  # (N, 384)
            text_tensor = torch.tensor(text_embs, dtype=torch.float32)
            # Align lengths
            n = min(self._last_hidden.shape[0], text_tensor.shape[0])
            fused = torch.cat([self._last_hidden[:n], text_tensor[:n]], dim=-1)
            logger.info(f"Fused features shape: {tuple(fused.shape)}")
            return fused
        except Exception as e:
            logger.warning(f"Fusion failed: {e}")
            return None
    
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
        Generate textual summary from audio transcripts and GNN scores.
        
        Args:
            audio_paths: List of shot audio file paths
            gnn_scores: GNN importance scores for each shot
            summary_type: Summary style (balanced, visual, audio, highlight)
            text_length: short/medium/long
            summary_format: bullet/structured/plain
            cache_dir: Optional cache directory for transcripts
            
        Returns:
            Formatted summary string
        """
        # Step 1: Transcribe audio
        logger.info(f"Transcribing {len(audio_paths)} audio files")
        transcriber = self.manager.get_whisper()
        transcripts = []
        
        for audio_path in audio_paths:
            audio_path_obj = Path(audio_path) if isinstance(audio_path, str) else audio_path
            if audio_path_obj.exists():
                try:
                    transcript = transcriber.transcribe(audio_path_obj, cache_dir)
                    # Filter out empty or garbage transcriptions
                    if transcript and len(transcript.strip()) > 3:
                        transcripts.append(transcript.strip())
                    else:
                        transcripts.append("")
                except Exception as e:
                    logger.error(f"Transcription failed for {audio_path_obj}: {e}")
                    transcripts.append("")
            else:
                transcripts.append("")
        
        valid_transcripts = sum(1 for t in transcripts if t)
        logger.info(f"Transcription complete ({valid_transcripts}/{len(transcripts)} valid)")
        
        # If too few valid transcriptions, warn user
        if valid_transcripts == 0:
            logger.warning("No valid transcriptions found - video may contain only music/noise")
            return "⚠️ No speech detected in video - unable to generate summary"
        
        # Optional: compute fused features for downstream use/logging
        _fused = self.fuse_hidden_with_text(transcripts)
        
        # Step 2: Generate summary
        logger.info(f"Generating {text_length} {summary_format} summary")
        summarizer = self.manager.get_summarizer()
        # Compute top-k from ratio
        n = len(gnn_scores)
        k_ratio = max(0.0, min(1.0, float(getattr(settings, "TOPK_RATIO", 0.15))))
        top_k = max(1, int(np.ceil(k_ratio * n))) if n > 0 else 1
        summary = summarizer.summarize(
            transcripts=transcripts,
            gnn_scores=gnn_scores,
            summary_type=summary_type,
            text_length=text_length,
            summary_format=summary_format,
            top_k=top_k
        )
        
        logger.info("Summary generation complete")
        return summary
    
    def process_video_pipeline(
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
        Complete end-to-end pipeline.
        
        Args:
            node_features: (N, 1536) shot features
            edge_index: (2, E) graph edges
            audio_paths: List of shot audio paths
            summary_type: Summary style
            text_length: short/medium/long
            summary_format: bullet/structured/plain
            cache_dir: Cache directory
            
        Returns:
            (gnn_scores, text_summary)
        """
        logger.info("Starting end-to-end inference pipeline")
        
        # Step 1: GNN scoring
        logger.info("Running GNN importance scoring")
        gnn_scores = self.predict_importance_scores(node_features, edge_index)
        logger.info(f"GNN scoring complete (probability range: {gnn_scores.min():.3f} - {gnn_scores.max():.3f})")
        
        # Step 2: Text summarization
        text_summary = self.generate_text_summary(
            audio_paths=audio_paths,
            gnn_scores=gnn_scores.tolist(),
            summary_type=summary_type,
            text_length=text_length,
            summary_format=summary_format,
            cache_dir=cache_dir
        )
        
        logger.info("Pipeline complete")
        return gnn_scores, text_summary
    
    def get_status(self) -> dict:
        """Get current service status"""
        return {
            "status": "ready",
            "models": self.manager.get_memory_stats(),
            "device": self.manager.get_device()
        }


# Global singleton instance
_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """
    Get or create global inference service instance.
    
    Returns:
        InferenceService singleton
    """
    global _service
    if _service is None:
        _service = InferenceService()
    return _service


def reset_inference_service():
    """Reset global service (useful for testing)"""
    global _service
    if _service is not None:
        _service.manager.clear_all()
    _service = None
    ModelManager.reset_instance()
