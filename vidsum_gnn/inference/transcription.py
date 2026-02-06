"""
Whisper-based audio transcription service.
Handles speech-to-text conversion with caching support.
"""
import torch
import librosa
import json
import time
import re
from pathlib import Path
from typing import Optional
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from requests.exceptions import ConnectionError, ChunkedEncodingError
from urllib3.exceptions import IncompleteRead

from vidsum_gnn.utils.logging import get_logger
from vidsum_gnn.core.config import settings

logger = get_logger(__name__)


def _looks_like_repetition_hallucination(text: str) -> bool:
    """Detect low-signal Whisper hallucinations like repeated single words.
    We keep this conservative to avoid dropping real speech.
    """
    import re

    t = (text or "").strip()
    if len(t) < 8:
        return False
    words = [w.lower() for w in re.findall(r"[a-zA-Z]{2,}", t)]
    if len(words) < 12:
        return False
    uniq = len(set(words))
    if uniq > 2:
        return False
    # Frequency of most common word
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    top = max(counts.values()) if counts else 0
    return (top / max(1, len(words))) >= 0.75


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
        self.whisper_language = (getattr(settings, "WHISPER_LANGUAGE", "auto") or "auto").strip().lower()
        
        logger.info(f"Loading Whisper model: {model_name}")
        
        # Load with retry logic for network interruptions
        self.processor = retry_with_backoff(
            lambda: WhisperProcessor.from_pretrained(
                model_name,
                resume_download=True,
                force_download=False
            )
        )
        
        self.model = retry_with_backoff(
            lambda: WhisperForConditionalGeneration.from_pretrained(
                model_name,
                resume_download=True,
                force_download=False
            ).to(self.device)
        )
        self.model.eval()

        if self.whisper_language != "auto":
            logger.info(f"✓ Whisper language forced: {self.whisper_language}")
        else:
            logger.info("✓ Whisper language: auto-detect")
        
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
            
            # Skip very short audio (usually UI noise)
            duration = len(audio) / 16000
            min_dur = float(getattr(settings, "WHISPER_MIN_DURATION_SEC", 0.5) or 0.5)
            if duration < min_dur:
                logger.debug(f"Skipping very short audio ({duration:.2f}s < {min_dur:.2f}s): {audio_path}")
                return ""

            # Skip near-silent segments. This prevents Whisper from hallucinating a repeated token
            # (we've observed 'Commission ...' style outputs) on low-energy audio.
            try:
                import numpy as _np

                rms = float(_np.sqrt(_np.mean(_np.square(audio)))) if len(audio) else 0.0
                thr = float(getattr(settings, "WHISPER_SILENCE_RMS_THRESHOLD", 0.0015) or 0.0015)
                if rms < thr:
                    logger.debug(
                        f"Skipping near-silent audio (rms={rms:.4f} < {thr:.4f}, {duration:.2f}s): {audio_path}"
                    )
                    return ""
            except Exception:
                # Never fail transcription due to RMS computation.
                pass
            
            # Process through Whisper
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            forced_decoder_ids = None
            if self.whisper_language and self.whisper_language != "auto":
                try:
                    # Force a particular language while keeping the task as transcription.
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=self.whisper_language,
                        task="transcribe",
                    )
                except Exception as e:
                    logger.warning(
                        f"Invalid WHISPER_LANGUAGE='{self.whisper_language}' (falling back to auto-detect): {e}"
                    )
                    forced_decoder_ids = None
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    max_new_tokens=128,
                    forced_decoder_ids=forced_decoder_ids,
                )
            
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            if _looks_like_repetition_hallucination(transcription):
                # Instead of fully discarding (which forces visual-only fallback),
                # keep a short deduped fragment so downstream summarization has some signal.
                tokens = transcription.split()
                dedup = []
                for t in tokens:
                    if not dedup or dedup[-1].lower() != t.lower():
                        dedup.append(t)
                fragment = " ".join(dedup[:12]).strip()
                if len(fragment.split()) < 3:
                    logger.warning(
                        f"Transcription looks like repetition hallucination; dropping: {transcription[:60]}"
                    )
                    return ""
                logger.warning(
                    f"Transcription looks like repetition hallucination; using deduped fragment: {fragment[:60]}"
                )
                transcription = fragment
            
            # Filter garbage transcriptions (all special chars, very short, etc.)
            if len(transcription) < 3:
                logger.debug(f"Transcription too short (<3 chars): {audio_path}")
                return ""
            
            # Check if transcription is mostly alphanumeric (filter garbage like "a b c d e f")
            alphanumeric_count = sum(1 for c in transcription if c.isalnum() or c.isspace())
            if alphanumeric_count / len(transcription) < 0.7:
                logger.warning(f"Transcription quality too low (mostly symbols): {transcription[:50]}")
                return ""
            
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

class TranscriptionService:
    """Thin wrapper used by the pipeline to keep a stable interface."""

    def __init__(self, model_name: str = "openai/whisper-base", device: Optional[str] = None):
        self.transcriber = WhisperTranscriber(model_name=model_name, device=device)

    def transcribe_audio(self, audio_path: Path | str, cache_dir: Optional[Path | str] = None) -> str:
        cache_path = Path(cache_dir) if cache_dir else None
        return self.transcriber.transcribe(Path(audio_path), cache_dir=cache_path)

    def transcribe_batch(self, audio_paths: list[Path | str], cache_dir: Optional[Path | str] = None) -> list[str]:
        cache_path = Path(cache_dir) if cache_dir else None
        paths = [Path(p) for p in audio_paths]
        return self.transcriber.batch_transcribe(paths, cache_dir=cache_path)
