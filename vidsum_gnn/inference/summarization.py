"""
Flan-T5 based text summarization service.
Generates formatted summaries from transcripts and GNN scores.

SUPPORTS:
- Summary Types: balanced, visual_priority, audio_priority, highlights
- Text Lengths: short (50-100 words), medium (100-200), long (200-400)
- Formats: bullet (• points), structured (with sections), plain (paragraphs)
"""
import torch
import numpy as np
import re
from typing import List, Optional, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)


class FlanT5Summarizer:
    """
    Advanced Flan-T5 for context-aware video summarization.
    
    Capabilities:
    - Content-aware summarization based on summary_type
    - Multiple output formats (bullet, structured, plain)
    - Length-controlled generation (short, medium, long)
    - Adaptive prompt engineering for different content types
    """
    
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
        self.max_input_chars = 2000
        
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
            transcripts: List of shot transcriptions (in original shot order)
            gnn_scores: GNN importance scores for each shot (0-1 range)
            summary_type: Type of summary
                - "balanced": Balanced visual + audio focus
                - "visual_priority": Focus on visual elements, scene descriptions
                - "audio_priority": Focus on dialogue, narration, spoken content
                - "highlights": Extract most exciting/important moments only
            text_length: 
                - "short": 50-100 words (2-3 sentences)
                - "medium": 100-200 words (4-6 sentences)
                - "long": 200-400 words (8-12 sentences)
            summary_format: 
                - "bullet": Bullet points (• format)
                - "structured": Sections with headers and organization
                - "plain": Natural paragraph format
            top_k: Number of top shots to include (auto-computed from ratio if possible)
            
        Returns:
            Formatted summary string
        """
        # Validate inputs
        if not transcripts or all(not t.strip() for t in transcripts):
            logger.warning("No valid transcripts provided, returning fallback")
            return self._generate_fallback_summary(summary_format, summary_type)
        
        if len(transcripts) != len(gnn_scores):
            logger.warning(f"Transcript-score mismatch: {len(transcripts)} vs {len(gnn_scores)}")
            # Pad or trim to match
            min_len = min(len(transcripts), len(gnn_scores))
            transcripts = transcripts[:min_len]
            gnn_scores = gnn_scores[:min_len]
        
        # Length parameters (in tokens, not words)
        length_map = {
            "short": (30, 70),
            "medium": (70, 150),
            "long": (150, 300)
        }
        min_length, max_length = length_map.get(text_length, (70, 150))
        
        # Select top-K shots based on GNN importance scores
        scores_arr = np.array(gnn_scores, dtype=np.float32)
        
        # Handle edge cases
        if scores_arr.max() > scores_arr.min():
            scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min() + 1e-6)
        else:
            scores_norm = np.ones_like(scores_arr) / len(scores_arr)
        
        # Select top-K while preserving original order
        num_shots = len(transcripts)
        top_k = min(top_k, num_shots)
        
        # Get indices of top-K shots
        top_indices = np.argsort(scores_norm)[::-1][:top_k]
        top_indices_sorted = np.sort(top_indices)  # Restore chronological order
        
        # Gather selected transcripts in original temporal order
        selected_transcripts = [
            transcripts[i] for i in top_indices_sorted 
            if i < len(transcripts) and transcripts[i].strip()
        ]
        
        if not selected_transcripts:
            logger.warning("No transcripts selected after filtering")
            return self._generate_fallback_summary(summary_format, summary_type)
        
        # Combine transcripts with summary type context
        context_prompt = self._get_context_prompt(summary_type)
        combined_text = " ".join(selected_transcripts)
        
        # Truncate if necessary
        if len(combined_text) > self.max_input_chars:
            combined_text = combined_text[:self.max_input_chars]
        
        # Generate summary with context
        summary_text = self._generate_summary(
            combined_text,
            context_prompt,
            min_length,
            max_length,
            summary_type
        )
        
        # Format output according to requested format
        formatted = self._format_summary(
            summary_text,
            summary_format,
            summary_type,
            text_length
        )
        
        logger.info(f"Generated {summary_type} {text_length} {summary_format} summary ({len(formatted)} chars)")
        return formatted
    
    def _get_context_prompt(self, summary_type: str) -> str:
        """
        Generate context-specific prompt prefix for Flan-T5.
        Guides the model to focus on specific aspects of the video.
        """
        prompts = {
            "balanced": (
                "Provide a balanced summary that covers both what you see and what you hear. "
                "Include key visual elements, scene changes, and important dialogue or narration."
            ),
            "visual_priority": (
                "Focus on visual elements and visual narrative. Describe what viewers see: "
                "scenes, objects, people, actions, visual effects, color schemes, and scene changes. "
                "De-emphasize dialogue unless it's essential."
            ),
            "audio_priority": (
                "Focus on audio content: dialogue, narration, sound effects, and what viewers hear. "
                "Prioritize spoken content and important audio elements. "
                "Only mention visuals if they directly support the audio narrative."
            ),
            "highlights": (
                "Extract only the most exciting, important, or memorable moments. "
                "Highlight turning points, key revelations, emotional moments, and climactic events. "
                "Be concise and focus on what's most impactful."
            )
        }
        return prompts.get(summary_type, prompts["balanced"])
    
    def _generate_summary(
        self,
        text: str,
        context_prompt: str,
        min_length: int,
        max_length: int,
        summary_type: str
    ) -> str:
        """
        Generate summary using Flan-T5 with context-aware prompting.
        
        Args:
            text: Combined transcript text
            context_prompt: Context-specific instruction
            min_length: Minimum generation length
            max_length: Maximum generation length
            summary_type: Type of summary (for logging)
        
        Returns:
            Generated summary text
        """
        # Build comprehensive prompt
        prompt = (
            f"{context_prompt}\n\n"
            f"Video Content:\n{text}\n\n"
            f"Summary:"
        )
        
        try:
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
                    no_repeat_ngram_size=3,
                    temperature=0.7,
                    top_p=0.9
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return summary if summary else "Unable to generate summary"
            
        except Exception as e:
            logger.error(f"Summary generation failed for {summary_type}: {e}")
            return "Unable to generate summary"
    
    def _format_summary(
        self,
        text: str,
        summary_format: str,
        summary_type: str,
        text_length: str
    ) -> str:
        """
        Format summary according to requested format.
        
        Args:
            text: Raw generated summary text
            summary_format: "bullet", "structured", or "plain"
            summary_type: For context in structured format
            text_length: For varying complexity
        
        Returns:
            Formatted summary string
        """
        if summary_format == "bullet":
            return self._to_bullet_points(text, text_length, summary_type)
        elif summary_format == "structured":
            return self._to_structured(text, summary_type, text_length)
        else:  # "plain"
            return self._to_plain(text)
    
    def _to_bullet_points(self, text: str, text_length: str, summary_type: str) -> str:
        """
        Convert summary to bullet point format.
        
        Intelligently splits sentences and creates digestible bullet points.
        """
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        # Vary number of bullets by length
        bullet_count = {
            "short": 3,
            "medium": 5,
            "long": 8
        }.get(text_length, 5)
        
        # Take the most relevant bullets
        bullets = sentences[:bullet_count]
        
        # Format as bullet list
        formatted_bullets = "\n".join([f"• {bullet.strip()}" for bullet in bullets])
        
        # Add header based on summary type
        header = {
            "balanced": "📹 Video Summary",
            "visual_priority": "👀 Visual Summary",
            "audio_priority": "🎧 Audio Summary",
            "highlights": "⭐ Highlights"
        }.get(summary_type, "📹 Summary")
        
        return f"{header}\n\n{formatted_bullets}"
    
    def _to_structured(self, text: str, summary_type: str, text_length: str) -> str:
        """
        Create a structured summary with organized sections.
        Perfect for detailed reviews and analysis.
        """
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        type_title = {
            "balanced": "Video Summary",
            "visual_priority": "Visual Analysis",
            "audio_priority": "Audio Summary",
            "highlights": "Key Highlights"
        }.get(summary_type, "Summary")
        
        if text_length == "short":
            return (
                f"## {type_title}\n\n"
                f"{sentences[0] if sentences else text}.\n\n"
                f"*[Auto-generated by VidSum GNN]*"
            )
        
        elif text_length == "long":
            result = f"## {type_title}\n\n"
            
            if sentences:
                result += f"### Overview\n{sentences[0]}.\n\n"
                
                if len(sentences) > 1:
                    mid_point = len(sentences) // 2
                    result += f"### Key Points\n"
                    result += "\n".join([f"- {s}" for s in sentences[1:mid_point]])
                    result += "\n\n"
                
                if len(sentences) > 2:
                    result += f"### Details\n"
                    result += "\n".join([f"- {s}" for s in sentences[len(sentences)//2:]])
                    result += "\n\n"
            
            result += f"*[Generated with Flan-T5 | Summary Type: {summary_type}]*"
            return result
        
        else:  # "medium"
            result = f"## {type_title}\n\n"
            result += f"{text}\n\n"
            result += f"*[VidSum GNN Summary]*"
            return result
    
    def _to_plain(self, text: str) -> str:
        """
        Return plain paragraph format with minimal formatting.
        """
        # Just clean up whitespace
        clean_text = " ".join(text.split())
        # Capitalize first letter if needed
        if clean_text and clean_text[0].islower():
            clean_text = clean_text[0].upper() + clean_text[1:]
        return clean_text
    
    def _generate_fallback_summary(self, summary_format: str, summary_type: str) -> str:
        """
        Generate fallback summary when no audio/transcripts available.
        Falls back to visual analysis from GNN scores.
        """
        fallback_texts = {
            "balanced": (
                "This is a visual summary of the video content. "
                "Key shots have been selected based on their importance scores from the video analysis. "
                "The most significant moments have been identified through visual and temporal analysis."
            ),
            "visual_priority": (
                "Visual Summary: The most visually important moments have been identified. "
                "These shots represent key scene changes, visual highlights, and important visual elements "
                "selected through deep visual analysis."
            ),
            "audio_priority": (
                "Audio Summary: No audio transcript was available for this video. "
                "Summary is based on visual importance analysis. For accurate audio-based summarization, "
                "ensure audio content is present in the video."
            ),
            "highlights": (
                "Key Highlights: The most impactful moments in the video have been identified "
                "through importance scoring. These represent the climactic and memorable moments."
            )
        }
        
        fallback_text = fallback_texts.get(summary_type, fallback_texts["balanced"])
        
        if summary_format == "bullet":
            return (
                f"📹 {summary_type.title()} Summary\n\n"
                f"• {fallback_text.split('.')[0]}.\n"
                f"• Key moments identified through visual analysis\n"
                f"• No audio transcript available"
            )
        elif summary_format == "structured":
            return (
                f"## {summary_type.title()} Summary\n\n"
                f"{fallback_text}\n\n"
                f"*[Visual Analysis Only - No Audio Available]*"
            )
        else:  # "plain"
            return fallback_text
