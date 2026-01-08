"""
Model Manager - Singleton for lazy loading and caching ML models.
Ensures efficient resource management across the application.
"""
import torch
from pathlib import Path
from typing import Optional

from vidsum_gnn.core.config import settings
from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Singleton manager for all inference models.
    Provides lazy loading and caching to optimize memory usage.
    """
    
    _instance: Optional['ModelManager'] = None
    
    def __init__(self):
        """Idempotent initializer for singleton usage"""
        # Allow multiple __init__ calls safely (e.g., via get_instance)
        if getattr(self, "_initialized", False):
            return
        
        self._gnn_model = None
        self._whisper = None
        self._text_embedder = None
        self._summarizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True
        
        logger.info(f"ModelManager initialized (device: {self._device})")
    
    @classmethod
    def get_instance(cls) -> 'ModelManager':
        """Get or create the singleton instance"""
        if cls._instance is None:
            # Use normal construction path; __init__ is idempotent
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton (useful for testing)"""
        if cls._instance is not None:
            cls._instance.clear_all()
        cls._instance = None
    
    def get_device(self) -> str:
        """Get the current compute device"""
        return self._device
    
    def get_gnn_model(self, checkpoint_path: Optional[Path] = None, in_dim: Optional[int] = None):
        """
        Get or load the GNN model.
        
        Args:
            checkpoint_path: Path to model checkpoint (optional)
            
        Returns:
            Loaded VidSumGNN model
        """
        if self._gnn_model is None:
            from vidsum_gnn.graph.model import VidSumGNN
            
            if checkpoint_path is None:
                # Prefer configured binary checkpoint; fallback to prior default
                configured = Path(settings.GNN_CHECKPOINT)
                checkpoint_path = configured if configured.exists() else Path(settings.MODEL_DIR) / "results" / "vidsumgnn_final.pt"
            
            logger.info(f"Loading GNN model from {checkpoint_path}")
            
            # Model configuration (should match training)
            model = VidSumGNN(
                in_dim=in_dim or 1536,  # ViT (768) + HuBERT (768)
                hidden_dim=getattr(settings, "GNN_HIDDEN_DIM", 512),
                num_heads=getattr(settings, "GNN_NUM_HEADS", 4),
                dropout=0.3
            )
            
            # Load checkpoint if exists
            if checkpoint_path.exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self._device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"✓ Loaded GNN weights from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}. Using untrained model.")
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}. Using untrained model.")
            
            model.to(self._device)
            model.eval()
            self._gnn_model = model
            
            logger.info("✓ GNN model ready")
        
        return self._gnn_model
    
    def get_whisper(self, model_name: str = "openai/whisper-base"):
        """
        Get or load Whisper transcription model.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            WhisperTranscriber instance
        """
        if self._whisper is None:
            from vidsum_gnn.inference.transcription import WhisperTranscriber
            self._whisper = WhisperTranscriber(model_name=model_name, device=self._device)
        return self._whisper
    
    def get_text_embedder(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Get or load text embedding model.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            TextEmbedder instance
        """
        if self._text_embedder is None:
            from vidsum_gnn.inference.text_embedding import TextEmbedder
            self._text_embedder = TextEmbedder(model_name=model_name, device=self._device)
        return self._text_embedder
    
    def get_summarizer(self, model_path: str = "google/flan-t5-base"):
        """
        Get or load Flan-T5 summarization model.
        
        Args:
            model_path: HuggingFace model path or local path
            
        Returns:
            FlanT5Summarizer instance
        """
        if self._summarizer is None:
            from vidsum_gnn.inference.summarization import FlanT5Summarizer
            
            # Use HuggingFace hub to avoid tokenizer version conflicts
            model_path = "google/flan-t5-base"
            logger.info(f"Using Flan-T5 model from HuggingFace Hub: {model_path}")
            
            self._summarizer = FlanT5Summarizer(model_path=model_path, device=self._device)
        return self._summarizer
    
    def clear_gnn_model(self):
        """Clear GNN model from memory"""
        if self._gnn_model is not None:
            del self._gnn_model
            self._gnn_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleared GNN model from memory")
    
    def clear_whisper(self):
        """Clear Whisper model from memory"""
        if self._whisper is not None:
            del self._whisper
            self._whisper = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleared Whisper model from memory")
    
    def clear_text_embedder(self):
        """Clear text embedder from memory"""
        if self._text_embedder is not None:
            del self._text_embedder
            self._text_embedder = None
            logger.info("Cleared text embedder from memory")
    
    def clear_summarizer(self):
        """Clear summarizer from memory"""
        if self._summarizer is not None:
            del self._summarizer
            self._summarizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleared summarizer from memory")
    
    def clear_all(self):
        """Clear all models from memory"""
        self.clear_gnn_model()
        self.clear_whisper()
        self.clear_text_embedder()
        self.clear_summarizer()
        logger.info("Cleared all models from memory")
    
    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics"""
        stats = {
            "gnn_loaded": self._gnn_model is not None,
            "whisper_loaded": self._whisper is not None,
            "text_embedder_loaded": self._text_embedder is not None,
            "summarizer_loaded": self._summarizer is not None,
            "device": self._device,
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            })
        
        return stats
