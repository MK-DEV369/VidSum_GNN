"""
Sentence-Transformers based text embedding service.
Converts text to dense semantic vectors.
"""
import numpy as np
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer

from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)


class TextEmbedder:
    """Sentence-Transformer for text embedding generation"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize text embedding model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Target device (cuda/cpu), auto-detected if None
        """
        self.device = device or "cuda"  # SentenceTransformer handles fallback
        self.model_name = model_name
        
        logger.info(f"Loading text embedder: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"✓ Text embedder loaded on {self.device}")
    
    def encode(self, text: str, cache_path: Optional[Path] = None) -> np.ndarray:
        """
        Encode text to embedding vector.
        
        Args:
            text: Input text string
            cache_path: Optional path to cache embedding
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        if cache_path and cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Save cache
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, embedding)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")
        
        return embedding
    
    def batch_encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode multiple texts to embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of embeddings (N, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings
