"""
Flan-T5 based text summarization service.
Generates formatted summaries from transcripts and GNN scores.

SUPPORTS:
- Summary Types: balanced, visual_priority, audio_priority, highlights
- Text Lengths: short (more detailed), medium (detailed), long (very detailed)
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


_PROFANITY_RE = re.compile(
    r"\b(" 
    r"fuck(?:er|ing|ed|s)?|shit(?:ty)?|bitch(?:es)?|cunt|pussy|dick(?:head)?|ass(?:hole)?|" 
    r"motherfucker|mf|bastard|slut|whore" 
    r")\b",
    flags=re.IGNORECASE,
)


def _sentence_case(line: str) -> str:
    t = (line or "").strip()
    if not t:
        return ""
    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    # Uppercase first alpha character
    for i, ch in enumerate(t):
        if ch.isalpha():
            if ch.islower():
                t = t[:i] + ch.upper() + t[i + 1 :]
            break
    return t


def _ensure_terminal_punct(line: str) -> str:
    t = (line or "").strip()
    if not t:
        return ""
    if t[-1] in ".!?":
        return t
    return t + "."


def _split_sentences(text: str) -> List[str]:
    """Split into sentences and remove very short/noisy fragments."""
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+|\n+", str(text))
    sentences: List[str] = []
    seen = set()
    for s in raw:
        s2 = _sentence_case(s)
        s2 = re.sub(r"\s+", " ", s2).strip()
        if len(s2) < 15:
            continue
        key = re.sub(r"\W+", "", s2).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        sentences.append(s2)
    return sentences


def _sanitize_profanity(text: str) -> str:
    """Remove/neutralize profanity without leaving the original tokens behind."""
    if not text:
        return ""
    t = str(text)
    # Replace profane tokens; do not preserve original word.
    t = _PROFANITY_RE.sub("", t)
    # Clean up double spaces / leftover punctuation spacing.
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([\.,!?])", r"\1", t)
    return t.strip()


def _sanitize_lines(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        t = _sanitize_profanity(line)
        t = _sentence_case(t)
        if len(t) < 12:
            continue
        cleaned.append(t)
    return cleaned


def _clean_transcript_text(text: str) -> str:
    """Heuristic cleanup for Whisper output.

    Removes boilerplate/license lines and collapses symbol-heavy gibberish.
    Designed to be conservative: if we can't improve it, we return a stripped version.
    """
    if not text:
        return ""
    t = str(text).replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""

    # Drop common boilerplate / licensing disclaimers that show up in some sources.
    bad_phrases = (
        "wikipedia",
        "creative commons",
        "attribution",
        "noncommercial",
        "no-derivatives",
        "unported license",
        "terms and conditions",
        "redistributed",
        "broadcast",
        "rewritten",
        "publication consideration",
        "material infringement",
    )
    sentences = re.split(r"(?<=[.!?])\s+", t)
    sentences = [s for s in sentences if s and not any(p in s.lower() for p in bad_phrases)]
    t = " ".join(sentences).strip()
    if not t:
        return ""

    # If the text is symbol-heavy, try to normalize punctuation and collapse repeats.
    # (Example: lots of WAA / punctuation / quote noise.)
    alnum = sum(1 for ch in t if ch.isalnum())
    ratio = alnum / max(1, len(t))
    if ratio < 0.55:
        t2 = re.sub(r"[^\w\s\.,!?-]", " ", t)
        t2 = re.sub(r"\s+", " ", t2).strip()
        # Collapse repeated short tokens: "WAA WAA WAA WAA" → "WAA"
        t2 = re.sub(r"\b(\w{1,6})\b(?:\s+\1\b){3,}", r"\1", t2, flags=re.IGNORECASE)
        t = t2 or t

    # Final sanity filter: require some real words.
    words = re.findall(r"[A-Za-z]{2,}", t)
    if len(words) < 3:
        return ""
    return t


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
        # Note: input is still token-truncated by the tokenizer; this just controls
        # a pre-trim to avoid extremely long strings.
        self.max_input_chars = 4000
        
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
                - "short": ~150-250 words (more detailed)
                - "medium": ~250-450 words (detailed)
                - "long": ~450-800 words (very detailed)
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
        summary_text = self._summarize_raw(
            transcripts=transcripts,
            gnn_scores=gnn_scores,
            summary_type=summary_type,
            text_length=text_length,
            top_k=top_k
        )

        if summary_text is None:
            return self._generate_fallback_summary(summary_format, summary_type)

        formatted = self._format_summary(summary_text, summary_format, summary_type, text_length)
        logger.info(f"✓ FINAL: {summary_type} {text_length} {summary_format} summary ({len(formatted)} chars)")
        return formatted

    def summarize_all_formats(
        self,
        transcripts: List[str],
        gnn_scores: List[float],
        summary_type: str = "balanced",
        text_length: str = "medium",
        top_k: int = 10,
    ) -> Dict[str, str]:
        """Generate bullet/structured/plain summaries in one model pass.

        This optimizes latency by generating the raw summary once and applying
        formatting transforms locally.
        """
        summary_text = self._summarize_raw(
            transcripts=transcripts,
            gnn_scores=gnn_scores,
            summary_type=summary_type,
            text_length=text_length,
            top_k=top_k
        )

        if summary_text is None:
            return {
                "bullet": self._generate_fallback_summary("bullet", summary_type),
                "structured": self._generate_fallback_summary("structured", summary_type),
                "plain": self._generate_fallback_summary("plain", summary_type),
            }

        return {
            "bullet": self._format_summary(summary_text, "bullet", summary_type, text_length),
            "structured": self._format_summary(summary_text, "structured", summary_type, text_length),
            "plain": self._format_summary(summary_text, "plain", summary_type, text_length),
        }

    def format_text(
        self,
        text: str,
        summary_format: str,
        summary_type: str = "balanced",
        text_length: str = "medium",
    ) -> str:
        """Format already-generated text into the requested output format."""
        if not text or not text.strip():
            return self._generate_fallback_summary(summary_format, summary_type)
        return self._format_summary(text, summary_format, summary_type, text_length)

    def _summarize_raw(
        self,
        transcripts: List[str],
        gnn_scores: List[float],
        summary_type: str,
        text_length: str,
        top_k: int,
    ) -> Optional[str]:
        """Return the raw (unformatted) summary text, or None if not possible."""
        # Validate inputs
        if not transcripts or all(not t.strip() for t in transcripts):
            logger.warning("No valid transcripts provided, returning fallback")
            return None

        # Clean transcripts to reduce Whisper gibberish / boilerplate.
        cleaned_transcripts = [_clean_transcript_text(t) for t in transcripts]
        if cleaned_transcripts and any(cleaned_transcripts):
            transcripts = cleaned_transcripts

        if len(transcripts) != len(gnn_scores):
            logger.warning(f"Transcript-score mismatch: {len(transcripts)} vs {len(gnn_scores)}")
            min_len = min(len(transcripts), len(gnn_scores))
            transcripts = transcripts[:min_len]
            gnn_scores = gnn_scores[:min_len]

        # These are *generation token* targets (not exact words). We intentionally
        # overshoot a bit since downstream formatting trims/splits.
        length_map = {
            "short": (160, 280),
            "medium": (280, 480),
            "long": (480, 768)
        }
        min_length, max_length = length_map.get(text_length, (280, 480))
        logger.info(f"✓ Length '{text_length}' → min_tokens={min_length}, max_tokens={max_length}")

        scores_arr = np.array(gnn_scores, dtype=np.float32)
        if scores_arr.size == 0:
            logger.warning("No GNN scores provided")
            return None

        if scores_arr.max() > scores_arr.min():
            scores_norm = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min() + 1e-6)
        else:
            scores_norm = np.ones_like(scores_arr) / max(len(scores_arr), 1)

        num_shots = len(transcripts)
        top_k = min(max(int(top_k), 1), num_shots)
        top_indices = np.argsort(scores_norm)[::-1][:top_k]
        top_indices_sorted = np.sort(top_indices)

        selected_transcripts = [
            transcripts[i] for i in top_indices_sorted
            if i < len(transcripts) and transcripts[i].strip()
        ]
        if not selected_transcripts:
            logger.warning("No transcripts selected after filtering")
            return None

        context_prompt = self._get_context_prompt(summary_type)
        combined_text = " ".join(selected_transcripts)
        if len(combined_text) > self.max_input_chars:
            combined_text = combined_text[:self.max_input_chars]

        summary_text = self._generate_summary(
            combined_text,
            context_prompt,
            min_length,
            max_length,
            summary_type,
            text_length
        )
        logger.info(f"✓ Raw summary: {len(summary_text)} chars")
        return summary_text
    
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
        summary_type: str,
        text_length: str
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
        word_targets = {
            "short": "150-250 words",
            "medium": "250-450 words",
            "long": "450-800 words",
        }
        target_words = word_targets.get(text_length, "250-450 words")
        prompt = (
            f"Write a clear, grammatical, family-friendly summary of the following video transcript in {target_words}. "
            f"Use complete sentences and coherent paragraphs. "
            f"Do NOT quote lines of dialogue verbatim and do NOT list random short phrases from the transcript. "
            f"If the transcript contains profanity or slurs, paraphrase without using any profanity.\n\n"
            f"{context_prompt}\n\n"
            f"Transcript:\n{text}\n\n"
            f"Summary:"
        )
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            # Encourage longer outputs for longer lengths.
            length_penalty = {"short": 0.9, "medium": 1.0, "long": 1.15}.get(text_length, 1.0)
            # Ensure min_length is always < max_length.
            safe_min_length = min(max(30, int(min_length)), int(max_length) - 1)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    min_length=safe_min_length,
                    max_length=int(max_length),
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=length_penalty
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
            return _sanitize_profanity(result)
        elif summary_format == "structured":
            result = self._to_structured(text, summary_type, text_length)
            logger.info(f"  → Structured format applied")
            return _sanitize_profanity(result)
        else:  # "plain"
            result = self._to_plain(text)
            logger.info(f"  → Plain format applied")
            return _sanitize_profanity(result)

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
        Convert summary to bullet point format.
        Intentionally avoids padding to large line counts (padding creates nonsense fragments).
        """
        sentences = _sanitize_lines(_split_sentences(text))

        target_lines = {
            "short": 8,
            "medium": 12,
            "long": 18,
        }.get(text_length, 10)

        bullets = sentences[:target_lines]
        if not bullets:
            return "• Unable to generate a clean bullet summary."

        formatted_bullets = "\n".join([f"• {_ensure_terminal_punct(b)}" for b in bullets])
        return formatted_bullets

    def _to_structured(self, text: str, summary_type: str, text_length: str) -> str:
        """
        Create a structured summary with organized sections.
        Removes the redundant top title (e.g., 'Video Summary') and the noisy 'Overview' label.
        """
        sentences = _sanitize_lines(_split_sentences(text))
        if not sentences:
            return "Unable to generate a clean structured summary."

        if text_length == "short":
            # 1 short paragraph, no headings.
            para = " ".join([_ensure_terminal_punct(s) for s in sentences[:4]])
            return para

        if text_length == "medium":
            # Light structure: key points only.
            key_points = sentences[:10]
            body = "\n".join([f"- {_ensure_terminal_punct(s)}" for s in key_points])
            return f"Key Points\n{body}"

        # long
        overview = " ".join([_ensure_terminal_punct(s) for s in sentences[:5]])
        key_points = sentences[5:15]
        details = sentences[15:25]

        result = overview.strip() + "\n\n"
        if key_points:
            result += "Key Points\n" + "\n".join([f"- {_ensure_terminal_punct(s)}" for s in key_points]) + "\n\n"
        if details:
            result += "Details\n" + "\n".join([f"- {_ensure_terminal_punct(s)}" for s in details])
        return result.strip()
    
    def _to_plain(self, text: str) -> str:
        """
        Return plain paragraph format with minimal formatting.
        """
        sentences = _sanitize_lines(_split_sentences(text))
        if not sentences:
            clean_text = " ".join(str(text).split())
            return _sentence_case(clean_text)

        # Plain summary: 2–3 coherent paragraphs, no bullets/headings.
        s = [_ensure_terminal_punct(x) for x in sentences]
        if len(s) <= 6:
            return " ".join(s)
        p1 = " ".join(s[:3])
        p2 = " ".join(s[3:6])
        p3 = " ".join(s[6:9]) if len(s) > 6 else ""
        out = (p1 + "\n\n" + p2 + ("\n\n" + p3 if p3 else "")).strip()
        return out
    
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
            return _sanitize_profanity(
                "\n".join(
                    [
                        f"• {fallback_text.split('.')[0].strip()}.",
                        "• Key moments identified through visual analysis.",
                        "• No audio transcript available.",
                    ]
                )
            )
        if summary_format == "structured":
            return _sanitize_profanity(
                f"Key Points\n- {fallback_text.strip()}\n- Visual analysis only (no transcript available)."
            )
        return _sanitize_profanity(fallback_text)
