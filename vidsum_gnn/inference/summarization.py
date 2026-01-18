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
import time
from typing import List, Optional, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from requests.exceptions import ConnectionError, ChunkedEncodingError
from urllib3.exceptions import IncompleteRead

from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)


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
        
        # Load with retry logic for network interruptions
        self.tokenizer = retry_with_backoff(
            lambda: AutoTokenizer.from_pretrained(
                model_path,
                resume_download=True,
                force_download=False
            )
        )
        
        self.model = retry_with_backoff(
            lambda: AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                resume_download=True,
                force_download=False
            ).to(self.device)
        )
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
        # Log input parameters for verification
        logger.info(f"\n=== SUMMARIZER CALLED ===")
        logger.info(f"Parameters: type={summary_type}, length={text_length}, format={summary_format}")
        logger.info(f"Transcripts: {len(transcripts)} shots, GNN scores: {len(gnn_scores)} scores")
        logger.info(f"Top-K: {top_k}")
        logger.info(f"=== END PARAMS ===")
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
        # Raised to encourage minimum line counts per length bucket.
        length_map = {
            "short": (90, 180),
            "medium": (180, 280),
            "long": (280, 480)
        }
        min_length, max_length = length_map.get(text_length, (180, 280))
        logger.info(f"✓ Length '{text_length}' → min_tokens={min_length}, max_tokens={max_length}")
        
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
        logger.info(f"✓ Raw summary: {len(summary_text)} chars, now formatting as '{summary_format}'...")
        formatted = self._format_summary(
            summary_text,
            summary_format,
            summary_type,
            text_length
        )
        
        logger.info(f"✓ FINAL: {summary_type} {text_length} {summary_format} summary ({len(formatted)} chars)")
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
        selected_prompt = prompts.get(summary_type, prompts["balanced"])
        logger.info(f"✓ Type '{summary_type}' → context prompt selected")
        return selected_prompt
    
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
        # Build aggressive prompt to force condensing and key point extraction
        # Use explicit instruction to avoid repeating the full text
        prompt = (
            f"Extract key points from this video transcript. Be concise and focus on main ideas. "
            f"Output only bullet points, each under 20 words. Do not repeat the entire transcript.\n\n"
            f"{context_prompt}\n\n"
            f"Transcript:\n{text}\n\n"
            f"Key Points:"
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
                    min_length=max(30, min_length // 2),  # Reduce min_length to prevent echoing
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    temperature=0.5,  # Lower temperature for more focused output
                    top_p=0.85,
                    length_penalty=0.5  # Penalize longer sequences to force condensing
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
        logger.info(f"  → Formatting: format={summary_format}, type={summary_type}, length={text_length}")
        
        if summary_format == "bullet":
            result = self._to_bullet_points(text, text_length, summary_type)
            logger.info(f"  → Bullet format applied")
            return result
        elif summary_format == "structured":
            result = self._to_structured(text, summary_type, text_length)
            logger.info(f"  → Structured format applied")
            return result
        else:  # "plain"
            result = self._to_plain(text)
            logger.info(f"  → Plain format applied")
            return result

    def _ensure_min_lines(self, sentences: List[str], min_lines: int, max_lines: Optional[int] = None) -> List[str]:
        """
        Ensure we have at least min_lines by splitting long sentences into clauses.
        Keeps order and caps at max_lines when provided.
        """
        lines: List[str] = [s for s in sentences if s]
        if len(lines) >= min_lines:
            return lines[:max_lines] if max_lines else lines

        for sentence in list(lines):
            if len(lines) >= min_lines:
                break
            parts = re.split(r",|;|-", sentence)
            for part in parts[1:]:
                clean = part.strip()
                if len(clean) > 3:
                    lines.append(clean)
                if len(lines) >= min_lines:
                    break

        if max_lines:
            return lines[:max_lines]
        return lines

    def _to_bullet_points(self, text: str, text_length: str, summary_type: str) -> str:
        """
        Convert summary to bullet point format with minimum line counts per length.
        """
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        target_lines = {
            "short": 5,
            "medium": 10,
            "long": 18  # aim for 15-20, cap at 18 to stay within range
        }.get(text_length, 5)

        sentences = self._ensure_min_lines(sentences, target_lines, target_lines)
        bullets = sentences[:target_lines]

        formatted_bullets = "\n".join([f"• {bullet.strip()}" for bullet in bullets])

        return f"\n{formatted_bullets}"

    def _to_structured(self, text: str, summary_type: str, text_length: str) -> str:
        """
        Create a structured summary with organized sections and enforced line counts.
        """
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        target_lines = {"short": 5, "medium": 10, "long": 18}.get(text_length, 10)
        sentences = self._ensure_min_lines(sentences, target_lines, target_lines)

        type_title = {
            "balanced": "Video Summary",
            "visual_priority": "Visual Analysis",
            "audio_priority": "Audio Summary",
            "highlights": "Key Highlights"
        }.get(summary_type, "Summary")

        if text_length == "short":
            body = "\n".join(sentences[:target_lines])
            return (
                f"{type_title}\n\n"
                f"{body}\n\n"
            )

        elif text_length == "long":
            result = f"{type_title}\n\n"

            overview = sentences[:2]
            key_points = sentences[2:10]
            details = sentences[10:target_lines]

            if overview:
                result += "Overview\n" + "\n".join(overview) + "\n\n"
            if key_points:
                result += "Key Points\n" + "\n".join([f"- {s}" for s in key_points]) + "\n\n"
            if details:
                result += "Details\n" + "\n".join([f"- {s}" for s in details]) + "\n\n"

            return result

        else:  # "medium"
            body = "\n".join([f"- {s}" for s in sentences[:target_lines]])
            return (
                f"{body}\n\n"
            )
    
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
