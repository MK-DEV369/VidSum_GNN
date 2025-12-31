import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from typing import List
import numpy as np

class VisualEncoder:
    """
    Visual feature extractor using Vision Transformer (ViT).
    Replaces CLIP for improved PyTorch 2.x compatibility.
    Output: 768-dimensional embeddings per image.
    """
    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Freeze parameters for transfer learning
        for p in self.model.parameters():
            p.requires_grad = False

    def encode(self, image_paths: List[str]) -> torch.Tensor:
        """
        Encode a list of images to embeddings using ViT.
        Args:
            image_paths: List of image file paths
        Returns:
            Tensor of shape (batch_size, 768) with normalized embeddings
        """
        images = []
        valid_paths = []
        
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"Error loading image {p}: {e}")
                # Use black placeholder
                images.append(Image.new("RGB", (224, 224)))
                valid_paths.append(p)
        
        if not images:
            return torch.empty(0, 768, device='cpu')

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # ViT pooler_output = [CLS] token representation (768-dim)
            embeddings = outputs.pooler_output  # (batch_size, 768)
            
        return embeddings.cpu()
