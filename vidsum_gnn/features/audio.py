import gc
import logging
from typing import List, Optional

import numpy as np
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor

logger = logging.getLogger(__name__)

class AudioEncoder:
    """
    Audio feature extractor using HuBERT (Meta).
    Replaces Wav2Vec2 for improved PyTorch 2.x compatibility.
    Output: 768-dimensional embeddings per audio window.
    """
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        device: Optional[str] = None,
        *,
        max_audio_seconds: float = 20.0,
        segment_seconds: float = 10.0,
        max_segments: int = 3,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Freeze parameters for transfer learning
        for p in self.model.parameters():
            p.requires_grad = False
        self.target_sr = 16000
        self.max_audio_seconds = float(max_audio_seconds)
        self.segment_seconds = float(segment_seconds)
        self.max_segments = int(max_segments)

        if self.max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be > 0")
        if self.segment_seconds <= 0:
            raise ValueError("segment_seconds must be > 0")
        if self.max_segments <= 0:
            raise ValueError("max_segments must be > 0")

    def _is_cuda_oom(self, exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "cuda out of memory" in msg or "cublas" in msg and "alloc" in msg

    def _split_into_segments(self, waveform_1d: torch.Tensor) -> List[torch.Tensor]:
        """Return a small list of representative segments (1D tensors)."""
        total_samples = int(waveform_1d.numel())
        if total_samples <= 0:
            return []

        max_samples = int(self.target_sr * self.max_audio_seconds)
        segment_samples = int(self.target_sr * self.segment_seconds)
        if segment_samples <= 0:
            segment_samples = min(max_samples, total_samples)

        # Short audio: just use it.
        if total_samples <= max_samples:
            return [waveform_1d]

        # Long audio: take up to max_segments segments across the clip.
        # This keeps memory bounded while still sampling start/middle/end context.
        segment_samples = min(segment_samples, max_samples)
        segment_samples = min(segment_samples, total_samples)
        if segment_samples <= 0:
            return []

        if self.max_segments == 1:
            start_indices = [0]
        elif self.max_segments == 2:
            start_indices = [0, max(0, total_samples - segment_samples)]
        else:
            mid = max(0, (total_samples - segment_samples) // 2)
            end = max(0, total_samples - segment_samples)
            start_indices = [0, mid, end]
            # If max_segments > 3, just truncate to 3 to avoid many forward passes.
            start_indices = start_indices[: min(self.max_segments, 3)]

        segments: List[torch.Tensor] = []
        for s in start_indices:
            e = min(s + segment_samples, total_samples)
            if e - s <= 0:
                continue
            segments.append(waveform_1d[s:e])
        return segments

    def _encode_segment(self, segment_1d: torch.Tensor, device: str) -> torch.Tensor:
        inputs = self.processor(
            segment_1d,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            pooled = torch.mean(hidden_states, dim=1)  # (1, 768)
        return pooled

    def encode(self, audio_paths: List[str]) -> torch.Tensor:
        """
        Encode a list of audio files to embeddings using HuBERT.
        Args:
            audio_paths: List of audio file paths (.wav, .mp3, etc.)
        Returns:
            Tensor of shape (batch_size, 768) - mean-pooled HuBERT hidden states
        """
        embeddings = []
        
        for p in audio_paths:
            try:
                # Load audio file
                waveform, sr = torchaudio.load(p)
                
                # Resample to 16kHz if needed
                if sr != self.target_sr:
                    resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                    waveform = resampler(waveform)
                
                # Convert stereo to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Squeeze to 1D and split/trim to keep memory bounded.
                input_values = waveform.squeeze()
                segments = self._split_into_segments(input_values)
                if not segments:
                    embeddings.append(torch.zeros(1, 768))
                    continue

                # Try on current device; if we hit CUDA OOM, permanently fall back to CPU.
                pooled_segments: List[torch.Tensor] = []
                try:
                    for seg in segments:
                        pooled_segments.append(self._encode_segment(seg, self.device))
                except RuntimeError as e:
                    if self.device.startswith("cuda") and self._is_cuda_oom(e):
                        logger.warning(
                            "HuBERT CUDA OOM while encoding %s; falling back to CPU for audio embeddings.",
                            p,
                        )
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        # Move model to CPU once; keep it there to avoid repeated OOM.
                        self.device = "cpu"
                        self.model = self.model.to("cpu")
                        pooled_segments = [self._encode_segment(seg, "cpu") for seg in segments]
                    else:
                        raise

                pooled = torch.mean(torch.cat(pooled_segments, dim=0), dim=0, keepdim=True)
                embeddings.append(pooled.cpu())

                # Cleanup
                del waveform, pooled, pooled_segments, segments, input_values
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                    
            except Exception as e:
                logger.warning("Error processing audio %s: %s", p, e)
                # Return zero vector on error
                embeddings.append(torch.zeros(1, 768))
        
        if not embeddings:
            return torch.empty(0, 768)
            
        return torch.cat(embeddings, dim=0)
