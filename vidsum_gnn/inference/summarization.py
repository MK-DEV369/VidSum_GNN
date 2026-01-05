"""
Flan-T5 based text summarization service.
Generates formatted summaries from transcripts and GNN scores.
"""
import torch
import numpy as np
import re
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)


class FlanT5Summarizer:
    """Flan-T5 for text summarization"""
    
    def __init__(
        self,
        model_path: str = "google/flan-t5-base",
        device: Optional[str] = None
    ):
        """
        Initialize Flan-T5 summarization model.
        
        Args:
            model_path: HuggingFace model path or local path
            device: Target device (cuda/cpu), auto-detected if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.max_input_chars = 1500
        
        logger.info(f"Loading Flan-T5 summarizer: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        logger.info(f"✓ Flan-T5 loaded on {self.device}")
    
    def summarize(
        self,
        transcripts: List[str],
        gnn_scores: List[float],
        summary_type: str = "balanced",
        text_length: str = "medium",
        summary_format: str = "bullet",
        top_k: int = 10
    ) -> str:
        """
        Generate text summary from transcripts and GNN scores.
        
        Args:
            transcripts: List of shot transcriptions
            gnn_scores: GNN importance scores for each shot
            summary_type: Type of summary (balanced, visual, audio, highlight)
            text_length: short (50-100 words), medium (100-200), long (200-400)
            summary_format: bullet, structured, or plain
            top_k: Number of top shots to include
            
        Returns:
            Formatted summary string
        """
        # Length parameters
        length_map = {
            "short": (30, 70),
            "medium": (70, 150),
            "long": (150, 300)
        }
        min_length, max_length = length_map.get(text_length, (70, 150))
        
        # Normalize scores
        scores_arr = np.array(gnn_scores)
        if scores_arr.max() > scores_arr.min():
            scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
        else:
            scores_norm = scores_arr
        
        # Select top-K shots
        indices = np.argsort(scores_norm)[::-1][:top_k]
        selected_transcripts = [transcripts[i] for i in indices if i < len(transcripts)]
        
        # Combine and truncate
        combined_text = " ".join(selected_transcripts)
        if len(combined_text) > self.max_input_chars:
            combined_text = combined_text[:self.max_input_chars]
        
        # Handle empty transcripts (no audio detected)
        if not combined_text.strip():
            return self._generate_fallback_summary(summary_format)
        
        # Generate summary
        summary_text = self._generate_summary(combined_text, min_length, max_length)
        
        # Format output
        return self._format_summary(
            summary_text,
            summary_format,
            summary_type,
            text_length
        )
    
    def _generate_summary(
        self,
        text: str,
        min_length: int,
        max_length: int
    ) -> str:
        """Generate summary using Flan-T5"""
        prompt = f"Summarize the following video content in {min_length}-{max_length} words:\n\n{text}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                min_length=min_length,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def _format_summary(
        self,
        text: str,
        summary_format: str,
        summary_type: str,
        text_length: str
    ) -> str:
        """Format summary according to requested format"""
        if summary_format == "bullet":
            return self._to_bullet_points(text, text_length)
        elif summary_format == "structured":
            return self._to_structured(text, summary_type, text_length)
        else:  # plain
            return text
    
    def _to_bullet_points(self, text: str, text_length: str) -> str:
        """Convert summary to bullet points"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Vary number of bullets by length
        num_bullets = {"short": 3, "medium": 5, "long": 8}.get(text_length, 5)
        bullets = sentences[:num_bullets]
        
        return "\n".join([f"• {sent}" for sent in bullets])
    
    def _to_structured(self, text: str, summary_type: str, text_length: str) -> str:
        """Create structured summary with sections"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if text_length == "short":
            return (
                f"**{summary_type.title()} Summary**\n\n"
                f"{'. '.join(sentences[:2])}.\n\n"
                "*Generated by VIDSUM-GNN*"
            )
        
        elif text_length == "long":
            intro = sentences[0] if sentences else text
            body = ". ".join(sentences[1:4]) if len(sentences) > 1 else ""
            conclusion = ". ".join(sentences[4:]) if len(sentences) > 4 else ""
            
            result = f"**{summary_type.title()} Summary**\n\n"
            result += f"**Overview:** {intro}.\n\n"
            if body:
                result += f"**Key Points:** {body}.\n\n"
            if conclusion:
                result += f"**Additional Details:** {conclusion}.\n\n"
            result += "*Generated by VIDSUM-GNN using Flan-T5*"
            return result
        
        else:  # medium
            return (
                f"**{summary_type.title()} Summary**\n\n"
                f"{text}\n\n"
                "*Generated by VIDSUM-GNN using Flan-T5*"
            )
    
    def _generate_fallback_summary(self, summary_format: str) -> str:
        """Generate fallback summary when no audio is detected"""
        fallbacks = {
            "bullet": (
                "• Visual highlights selected by importance scoring\n"
                "• Key moments detected through shot analysis\n"
                "• No audio transcript available"
            ),
            "structured": (
                "Summary: Video summary generated from visual shot importance analysis. "
                "Audio transcription not available."
            ),
            "plain": (
                "This is a visual summary generated from shot importance analysis. "
                "The most important shots have been selected based on visual features "
                "and their relationships in the video."
            )
        }
        return fallbacks.get(summary_format, fallbacks["plain"])
