import torch
import torchaudio
from transformers import HubertModel, HubertProcessor
from typing import List
import numpy as np

class AudioEncoder:
    """
    Audio feature extractor using HuBERT (Meta).
    Replaces Wav2Vec2 for improved PyTorch 2.x compatibility.
    Output: 768-dimensional embeddings per audio window.
    """
    def __init__(self, model_name: str = "facebook/hubert-base-ls960", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = HubertProcessor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Freeze parameters for transfer learning
        for p in self.model.parameters():
            p.requires_grad = False
        self.target_sr = 16000

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
                
                # Squeeze to 1D
                input_values = waveform.squeeze()
                
                # Process through HuBERT
                inputs = self.processor(
                    input_values, 
                    sampling_rate=self.target_sr, 
                    return_tensors="pt", 
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # last_hidden_state: (batch, seq_len, hidden_size=768)
                    # Mean pool over temporal dimension
                    hidden_states = outputs.last_hidden_state
                    pooled = torch.mean(hidden_states, dim=1)  # (batch, 768)
                    embeddings.append(pooled)
                    
            except Exception as e:
                print(f"Error processing audio {p}: {e}")
                # Return zero vector on error
                embeddings.append(torch.zeros(1, 768, device=self.device))
        
        if not embeddings:
            return torch.empty(0, 768, device='cpu')
            
        return torch.cat(embeddings, dim=0).cpu()
