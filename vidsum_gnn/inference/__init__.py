"""
Inference module for video summarization models.

This module provides:
- Model loading and management
- Audio transcription (Whisper)
- Text embedding (Sentence-Transformers)
- Text summarization (Flan-T5)
- End-to-end inference pipeline
"""

from vidsum_gnn.inference.service import get_inference_service, InferenceService
from vidsum_gnn.inference.model_manager import ModelManager

__all__ = [
    "get_inference_service",
    "InferenceService",
    "ModelManager",
]
