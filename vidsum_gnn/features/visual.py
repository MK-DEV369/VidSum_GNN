import torch
import gc
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from typing import List
from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)

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
        embeddings = []
        batch_size = 4  # Process 4 images at a time
        
        for batch_idx in range(0, len(image_paths), batch_size):
            batch_end = min(batch_idx + batch_size, len(image_paths))
            batch_paths = image_paths[batch_idx:batch_end]
            
            images = []
            for p in batch_paths:
                try:
                    with Image.open(p) as img:
                        images.append(img.convert("RGB"))
                except Exception as e:
                    logger.warning(f"Error loading image {p}: {e}")
                    images.append(Image.new("RGB", (224, 224)))
            
            if not images:
                continue

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.pooler_output
                if batch_embeddings is None:
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu())
            
            # Cleanup
            del inputs, outputs, images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        if not embeddings:
            return torch.empty(0, 768)
            
        return torch.cat(embeddings, dim=0)
