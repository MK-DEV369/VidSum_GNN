"""
Sentence-Transformers based text embedding service.
Converts text to dense semantic vectors.
"""
import numpy as np
import time
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer
from requests.exceptions import ConnectionError, ChunkedEncodingError
from urllib3.exceptions import IncompleteRead

from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)


def retry_with_backoff(func, max_retries=3, initial_delay=2):
    """
    Retry a function with exponential backoff on network errors.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (doubles each retry)
    
    Returns:
        Function result if successful
        
    Raises:
        Last exception if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except (ConnectionError, ChunkedEncodingError, IncompleteRead, OSError) as e:
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} retry attempts failed")
                raise
            
            delay = initial_delay * (2 ** attempt)
            logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}...")
            logger.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    
    raise RuntimeError("Retry logic failed unexpectedly")


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
        
        # Load with retry logic for network interruptions
        self.model = retry_with_backoff(
            lambda: SentenceTransformer(model_name, device=self.device)
        )
        
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
