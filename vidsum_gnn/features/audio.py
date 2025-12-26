import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import List
import numpy as np

class AudioEncoder:
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.target_sr = 16000

    def encode(self, audio_paths: List[str]) -> torch.Tensor:
        """
        Encode a list of audio files to embeddings.
        Returns Tensor of shape (batch_size, 768) - Wav2Vec2 base is 768d, not 512d usually, but we'll check.
        Actually Wav2Vec2-base hidden size is 768. The spec said 128-512, but we'll use what the model gives.
        We will pool the sequence output to get a single vector per shot.
        """
        embeddings = []
        
        for p in audio_paths:
            try:
                waveform, sr = torchaudio.load(p)
                if sr != self.target_sr:
                    resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                    waveform = resampler(waveform)
                
                # Mix to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Squeeze to 1D
                input_values = waveform.squeeze()
                
                # Chunk if too long (max 30s shot usually, but let's be safe)
                # Wav2Vec2 can handle long inputs but memory is an issue.
                # We'll just take the first 30s or so if it's huge, or let the processor handle it.
                # For simplicity, we process the whole thing and mean pool.
                
                inputs = self.processor(input_values, sampling_rate=self.target_sr, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # last_hidden_state: (batch, seq_len, hidden_size)
                    # Mean pool over time
                    hidden_states = outputs.last_hidden_state
                    pooled = torch.mean(hidden_states, dim=1)
                    embeddings.append(pooled)
                    
            except Exception as e:
                print(f"Error processing audio {p}: {e}")
                # Zero vector
                embeddings.append(torch.zeros(1, 768).to(self.device))
        
        if not embeddings:
            return torch.empty(0, 768)
            
        return torch.cat(embeddings, dim=0).cpu()
