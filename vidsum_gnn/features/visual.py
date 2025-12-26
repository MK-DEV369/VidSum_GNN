import torch
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from typing import List
import numpy as np

class VisualEncoder:
    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, image_paths: List[str]) -> torch.Tensor:
        """
        Encode a list of images to embeddings.
        Returns Tensor of shape (batch_size, 768).
        """
        images = []
        for p in image_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"Error loading image {p}: {e}")
                # Placeholder black image
                images.append(Image.new("RGB", (224, 224)))

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use pooler_output (CLS token)
            embeddings = outputs.pooler_output
            
        return embeddings.cpu()
