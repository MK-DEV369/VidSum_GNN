"""
Whisper-based audio transcription service.
Handles speech-to-text conversion with caching support.
"""
import torch
import librosa
import json
from pathlib import Path
from typing import Optional
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)


class WhisperTranscriber:
    """Whisper ASR for audio-to-text transcription"""
    
    def __init__(self, model_name: str = "openai/whisper-base", device: Optional[str] = None):
        """
        Initialize Whisper transcription model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Target device (cuda/cpu), auto-detected if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        logger.info(f"Loading Whisper model: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logger.info(f"✓ Whisper loaded on {self.device}")
    
    def transcribe(self, audio_path: Path, cache_dir: Optional[Path] = None) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
            cache_dir: Optional directory for caching transcripts
            
        Returns:
            Transcribed text string
        """
        audio_path = Path(audio_path)
        
        # Check cache
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{audio_path.stem}_transcript.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)["text"]
                except Exception as e:
                    logger.warning(f"Failed to load cache {cache_file}: {e}")
        
        # Validate file
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return ""
        
        if audio_path.stat().st_size == 0:
            logger.warning(f"Audio file is empty: {audio_path}")
            return ""
        
        try:
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            
            if len(audio) == 0:
                logger.warning(f"Audio loaded but empty: {audio_path}")
                return ""
            
            # Process through Whisper
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    max_new_tokens=128
                )
            
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # Cache result
            if cache_dir and cache_file:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump({"text": transcription}, f)
                except Exception as e:
                    logger.warning(f"Failed to write cache {cache_file}: {e}")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription error for {audio_path}: {e}")
            return ""
    
    def batch_transcribe(
        self,
        audio_paths: list[Path],
        cache_dir: Optional[Path] = None
    ) -> list[str]:
        """
        Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            cache_dir: Optional cache directory
            
        Returns:
            List of transcriptions (empty string for failed files)
        """
        return [self.transcribe(path, cache_dir) for path in audio_paths]
