"""Flan-T5 summarizer.

Generates summaries from shot transcripts and GNN importance scores.
"""
import torch
import numpy as np
import re
import hashlib
import time
from collections import Counter
from typing import List, Optional, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from requests.exceptions import ConnectionError, ChunkedEncodingError
from urllib3.exceptions import IncompleteRead

from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)


_BOILERPLATE_PHRASES = (
    "associated press",
    "the associated press",
    "reuters",
    "all rights reserved",
    "may not be published",
    "may not be broadcast",
    "may not be rewritten",
    "may not be redistributed",
    "express written permission",
    "this material may not be",
    "copyright act",
    "strictly prohibited",
)


_TRANSCRIPT_DROP_PHRASES = _BOILERPLATE_PHRASES + (
    "wikipedia",
    "creative commons",
    "attribution",
    "noncommercial",
    "no-derivatives",
    "unported license",
    "terms and conditions",
    "publication consideration",
    "material infringement",
)


def _text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


def _boilerplate_hit_count(text: str) -> int:
    if not text:
        return 0
    tl = str(text).lower()
    return sum(1 for p in _BOILERPLATE_PHRASES if p in tl)


def _preview(text: str, limit: int = 220) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", str(text)).strip()
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + "…"


def _repetition_stats(text: str) -> dict:
    """Lightweight repetition metrics to detect ASR/model looping."""
    t = str(text or "")
    words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", t)]
    total = len(words)
    if total == 0:
        return {
            "total": 0,
            "uniq": 0,
            "diversity": 0.0,
            "top_ratio": 0.0,
            "top_word": "",
            "top_bigram_count": 0,
            "top_bigram_ratio": 0.0,
        }

    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    uniq = len(counts)
    diversity = uniq / max(1, total)
    top_ratio = top_count / max(1, total)

    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(total - 1)]
    if bigrams:
        bc = Counter(bigrams)
        _bg, bg_count = bc.most_common(1)[0]
        bg_ratio = bg_count / max(1, len(bigrams))
    else:
        bg_count = 0
        bg_ratio = 0.0

    return {
        "total": total,
        "uniq": uniq,
        "diversity": float(diversity),
        "top_ratio": float(top_ratio),
        "top_word": str(top_word),
        "top_bigram_count": int(bg_count),
        "top_bigram_ratio": float(bg_ratio),
    }


def _looks_like_looping_hallucination(text: str) -> bool:
    """Detect low-signal looping text (ASR or model) and prefer fallback over nonsense."""
    t = str(text or "").strip()
    if not t:
        return False

    # Excessive punctuation loops like ".............." or "____".
    if len(t) >= 60:
        dots = t.count(".")
        underscores = t.count("_")
        if (dots / max(1, len(t))) >= 0.35 or underscores >= 12:
            return True

    stats = _repetition_stats(t)
    total = int(stats["total"])
    if total < 12:
        return False

    diversity = float(stats["diversity"])
    top_ratio = float(stats["top_ratio"])
    top_word = str(stats["top_word"])
    bg_count = int(stats["top_bigram_count"])
    bg_ratio = float(stats["top_bigram_ratio"])

    # Common junk tokens we see in Whisper hallucinations.
    junk_tokens = {
        "commission",
        "importantly",
        "everyone",
        "times",
        "subscribe",
        "subscribed",
        "like",
        "share",
        "thanks",
        "thank",
    }

    # Hard drop: dominated by known junk.
    if top_word in junk_tokens and top_ratio >= 0.22:
        return True

    # Looping bigrams like "short short" or "the the" repeated many times.
    if total >= 18 and bg_count >= 5 and bg_ratio >= 0.22:
        return True

    # Low diversity + one token dominating.
    if total >= 20 and (diversity <= 0.35) and (top_ratio >= 0.30):
        return True

    # Longer sequences that still look repetitive.
    if total >= 30 and ((diversity < 0.18) or (top_ratio >= 0.45 and diversity < 0.35)):
        return True

    return False


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
    t = re.sub(r"\s+", " ", t).strip()
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
        # Allow shorter fragments (sports commentary is often very short).
        if len(s2) < 10:
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
    t = _PROFANITY_RE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([\.,!?])", r"\1", t)
    return t.strip()


def _sanitize_lines(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        t = _sanitize_profanity(line)
        tl = t.lower()
        if any(p in tl for p in _TRANSCRIPT_DROP_PHRASES):
            continue
        t = _sentence_case(t)
        # Keep short but meaningful lines (e.g., sports callouts / lecture keywords).
        if len(t) < 4:
            continue
        cleaned.append(t)
    return cleaned


def _clean_transcript_text(text: str) -> str:
    """Heuristic cleanup for Whisper output."""
    if not text:
        return ""
    t = str(text).replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", t)
    sentences = [s for s in sentences if s and not any(p in s.lower() for p in _TRANSCRIPT_DROP_PHRASES)]

    # De-loop common ASR repetition patterns by de-duplicating repeated short sentences,
    # rather than dropping the whole transcript (which often starves summarization).
    if sentences:
        garbage_words = {"commission", "times", "everyone", "importantly"}
        key_counts: Counter[str] = Counter()
        filtered: List[str] = []
        for s in sentences:
            w = re.findall(r"[A-Za-z]{2,}", s.lower())
            if not w:
                continue
            # Drop standalone low-signal tokens that show up frequently in noisy ASR.
            if len(w) == 1 and w[0] in garbage_words:
                continue
            key = " ".join(w[:6])
            # Allow fewer repeats for very short keys.
            limit = 1 if len(key.split()) <= 3 else 2
            if key_counts[key] >= limit:
                continue
            key_counts[key] += 1
            filtered.append(s)
        sentences = filtered

    t = " ".join(sentences).strip()
    if not t:
        return ""

    # If ASR output is essentially a short loop of the same junk token
    # (common on noisy / music / near-silent segments), drop it early.
    # This specifically fixes cases like: "Commission Commission Commission" which
    # are too short to trip the >=12 word repetition filter below.
    words_short = re.findall(r"[A-Za-z]{2,}", t)
    if words_short:
        lw_short = [w.lower() for w in words_short]
        total_short = len(lw_short)
        counts_short = Counter(lw_short)
        top_word, top_count = counts_short.most_common(1)[0]
        top_ratio_short = top_count / max(1, total_short)
        # Only be aggressive for known junk tokens.
        junk_tokens = {
            "commission",
            "times",
            "everyone",
            "importantly",
            "subscribe",
            "subscribed",
            "like",
            "share",
            "thanks",
            "thank",
        }
        if total_short >= 3 and top_word in junk_tokens and top_ratio_short >= 0.80:
            return ""

    # Collapse extreme repeated-word hallucinations rather than dropping the entire transcript.
    # Example: "down down down down down" -> "down down down".
    def _collapse_word_runs(m: re.Match) -> str:
        w = m.group(1)
        return f"{w} {w} {w}"

    t = re.sub(r"\b([A-Za-z]{2,})(?:\s+\1){2,}\b", _collapse_word_runs, t, flags=re.IGNORECASE)

    # Drop repeated-token hallucinations.
    words = re.findall(r"[A-Za-z]{2,}", t)
    if words:
        lw = [w.lower() for w in words]
        total = len(lw)
        uniq = len(set(lw))

        counts = Counter(lw)
        top = max(counts.values()) if counts else 0

        # Extremely low lexical diversity.
        if total >= 12 and (uniq <= 2) and (top / max(1, total)) >= 0.75:
            return ""

        # Short-loop junk that survives the thresholds above.
        # Keep this narrow to avoid dropping legitimate emphasis ("go go go").
        if total >= 3 and uniq <= 2:
            top_word = counts.most_common(1)[0][0] if counts else ""
            if top_word in {"commission", "times", "everyone", "importantly"} and (top / max(1, total)) >= 0.80:
                return ""

        # Broader repetition patterns that still slip past the check above.
        # Examples: lots of 'down', 'times', 'is' mixed with a few tokens.
        diversity = uniq / max(1, total)
        top_ratio = top / max(1, total)
        if total >= 30 and ((diversity < 0.18) or (top_ratio >= 0.45 and diversity < 0.35)):
            return ""

        # Bigram looping (catches cases like: "short short short ..." that don't hit thresholds above).
        if total >= 18:
            bigrams = [f"{lw[i]} {lw[i + 1]}" for i in range(total - 1)]
            if bigrams:
                bg_counts = Counter(bigrams)
                _bg, bg_top = bg_counts.most_common(1)[0]
                bg_ratio = bg_top / max(1, len(bigrams))
                if bg_top >= 5 and bg_ratio >= 0.22:
                    return ""

    alnum = sum(1 for ch in t if ch.isalnum())
    ratio = alnum / max(1, len(t))
    if ratio < 0.55:
        t2 = re.sub(r"[^\w\s\.,!?-]", " ", t)
        t2 = re.sub(r"\s+", " ", t2).strip()
        t = t2 or t

    words2 = re.findall(r"[A-Za-z]{2,}", t)
    if len(words2) < 2:
        # Allow single-keyword transcripts (common in sparse ASR) but drop obvious junk.
        w = (words2[0].lower() if words2 else "")
        junk = {
            "commission",
            "times",
            "everyone",
            "importantly",
            "subscribe",
            "like",
            "share",
            "thanks",
            "thank",
        }
        if not w or w in junk:
            return ""
    return t

def _split_inline_bullets(text: str) -> List[str]:
    """Split text that already contains inline bullet markers into separate items."""
    if not text:
        return []
    t = str(text).strip()
    if "\n" in t:
        return []
    if t.count("•") < 2 and t.count("-") < 2:
        return []

    if t.count("•") >= 2:
        parts = [p.strip() for p in re.split(r"\s*•\s+", t) if p.strip()]
        return parts

    parts = [p.strip() for p in re.split(r"\s+-\s+", t) if p.strip()]
    return parts

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
    """Generate summaries using Flan-T5."""
    
    def __init__(
        self,
        model_path: str = "google/flan-t5-base",
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.max_input_chars = 4000
        
        logger.info(f"Loading Flan-T5 summarizer: {model_path}")
        
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
        """Generate a formatted summary."""
        text_length_raw = text_length
        text_length = self._normalize_text_length(text_length)
        logger.info(f"\n=== SUMMARIZER CALLED ===")
        if str(text_length_raw).strip().lower() != str(text_length).strip().lower():
            logger.info(
                f"Parameters: type={summary_type}, length={text_length_raw}→{text_length}, format={summary_format}"
            )
        else:
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
        try:
            raw_words = len(re.findall(r"[A-Za-z]{2,}", str(summary_text or "")))
            formatted_words = len(re.findall(r"[A-Za-z]{2,}", formatted))
            logger.info(
                f"✓ Output stats: raw_words={raw_words} formatted_words={formatted_words} format={summary_format}"
            )
        except Exception:
            pass
        logger.info(f"✓ FINAL: {summary_type} {text_length} {summary_format} summary ({len(formatted)} chars)")
        return formatted

    def summarize_all_formats(
        self,
        transcripts: List[str],
        gnn_scores: List[float],
        summary_type: str = "balanced",
        text_length: str = "medium",
        top_k: int = 10,
        video_path: Optional[str] = None,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Generate bullet/structured/plain summaries in one model pass.

        This optimizes latency by generating the raw summary once and applying
        formatting transforms locally.
        """
        fmt_in = formats or ["bullet", "structured", "plain"]
        fmt_clean: List[str] = []
        for f in fmt_in:
            if not f:
                continue
            ff = str(f).strip().lower()
            if ff in {"bullet", "structured", "plain"}:
                fmt_clean.append(ff)
        if not fmt_clean:
            fmt_clean = ["bullet"]

        text_length = self._normalize_text_length(text_length)
        summary_text = self._summarize_raw(
            transcripts=transcripts,
            gnn_scores=gnn_scores,
            summary_type=summary_type,
            text_length=text_length,
            top_k=top_k
        )

        if summary_text is None:
            # Try Gemini fallback if available and a video path is provided.
            if video_path:
                try:
                    from vidsum_gnn.inference.gemini_fallback import get_gemini_summarizer

                    gemini = get_gemini_summarizer()
                    if gemini.is_available():
                        g_sum, _meta = gemini.summarize_video_from_path(
                            video_path,
                            summary_type=summary_type,
                            text_length=text_length,
                            summary_format="plain",
                        )
                        if g_sum:
                            logger.info("✓ Gemini fallback summary used")
                            summary_text = g_sum
                except Exception as e:
                    logger.warning(f"Gemini fallback failed: {e}")

            if summary_text is None:
                return {
                    fmt: self._generate_fallback_summary(fmt, summary_type)
                    for fmt in fmt_clean
                }

        return {
            fmt: self._format_summary(summary_text, fmt, summary_type, text_length)
            for fmt in fmt_clean
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
        text_length = self._normalize_text_length(text_length)
        return self._format_summary(text, summary_format, summary_type, text_length)

    def _normalize_text_length(self, text_length: str) -> str:
        """Normalize UI/API variants to internal values."""
        tl = str(text_length or "").strip().lower()
        if tl in ("small", "short"):
            return "short"
        if tl == "medium":
            return "medium"
        if tl == "long":
            return "long"
        if tl:
            logger.warning(f"Unknown text_length='{text_length}', defaulting to 'medium'")
        return "medium"

    def _summarize_raw(
        self,
        transcripts: List[str],
        gnn_scores: List[float],
        summary_type: str,
        text_length: str,
        top_k: int,
    ) -> Optional[str]:
        """Return the raw (unformatted) summary text, or None if not possible."""
        text_length = self._normalize_text_length(text_length)
        if not transcripts or all(not t.strip() for t in transcripts):
            logger.warning("No valid transcripts provided, returning fallback")
            return None

        nonempty_before = sum(1 for t in transcripts if t and t.strip())
        logger.info(
            f"Summarizer input: shots={len(transcripts)}, nonempty={nonempty_before}, "
            f"scores={len(gnn_scores)}"
        )

        cleaned_transcripts = [_clean_transcript_text(t) for t in transcripts]
        if cleaned_transcripts and any(cleaned_transcripts):
            transcripts = cleaned_transcripts

        nonempty_after = sum(1 for t in transcripts if t and t.strip())
        boiler_hits = sum(1 for t in transcripts if _boilerplate_hit_count(t) > 0)
        logger.info(
            f"After transcript cleaning: nonempty={nonempty_after} (was {nonempty_before}), "
            f"boilerplate_hits={boiler_hits}"
        )

        if len(transcripts) != len(gnn_scores):
            logger.warning(f"Transcript-score mismatch: {len(transcripts)} vs {len(gnn_scores)}")
            min_len = min(len(transcripts), len(gnn_scores))
            transcripts = transcripts[:min_len]
            gnn_scores = gnn_scores[:min_len]

        # Generation length targets are in *model tokens*.
        # Keep these modest: forcing very long generations increases runtime and
        # increases hallucination/loop risk when transcripts are short.
        length_map = {
            "short": (64, 160),
            "medium": (160, 260),
            "long": (320, 520),
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
            # Common failure mode: the highest-scoring shots are purely visual, but a few lower-scoring
            # shots still have usable ASR. Expand the selection beyond top_k until we find some text.
            logger.warning(
                "No transcripts selected from top-K shots; expanding search for any non-empty transcripts"
            )
            ranked = np.argsort(scores_norm)[::-1]
            expanded: List[str] = []
            for idx in ranked:
                i = int(idx)
                if i < 0 or i >= len(transcripts):
                    continue
                t = transcripts[i]
                if t and t.strip():
                    expanded.append(t)
                if len(expanded) >= top_k:
                    break
            selected_transcripts = expanded

        if not selected_transcripts:
            logger.warning("No transcripts available after expanded search")
            return None

        if logger.logger.isEnabledFor(10):
            # DEBUG-only: show which indices were selected and short previews.
            try:
                logger.debug(f"Top indices (sorted): {top_indices_sorted.tolist()}")
            except Exception:
                logger.debug("Top indices (sorted) computed")
            for j, idx in enumerate(top_indices_sorted[: min(6, len(top_indices_sorted))]):
                if idx < len(transcripts) and transcripts[idx].strip():
                    logger.debug(f"Selected[{j}] idx={int(idx)} txt='{_preview(transcripts[idx], 140)}'")

        context_prompt = self._get_context_prompt(summary_type)
        combined_text = " ".join(selected_transcripts)

        combined_words = re.findall(r"[A-Za-z]{2,}", combined_text)
        signal_words = len(combined_words)

        # Cricket/sports/lectures can yield sparse keyword-like ASR; producing *some* text output is
        # better than the scary visual-only fallback when we have any usable words.
        if signal_words == 0:
            logger.warning("No transcript words found after cleaning; returning fallback")
            return None

        # If the selected transcript itself looks like a looping hallucination, do not try to expand it
        # into a long summary. Prefer the safer fallback.
        if _looks_like_looping_hallucination(combined_text):
            stats = _repetition_stats(combined_text)
            logger.warning(
                "Transcript looks repetitive/low-signal; using fallback | "
                f"words={stats.get('total')} uniq={stats.get('uniq')} top={stats.get('top_word')} "
                f"top_ratio={stats.get('top_ratio'):.2f} bg_top={stats.get('top_bigram_count')}"
            )
            return None

        if signal_words < 8:
            logger.warning(
                f"Very low transcript signal (words={signal_words}); returning transcript fragments instead of fallback"
            )
            return combined_text[: self.max_input_chars].strip()

        # Calibrate generation length based on signal strength to avoid hallucinations but respect user length.
        if signal_words < 25:
            logger.warning(
                f"Low transcript signal (words={signal_words}); clamping to short to avoid hallucinations"
            )
            min_length, max_length = length_map.get("short", (64, 160))
            context_prompt = (
                context_prompt
                + " Transcript fragments are sparse/noisy. Summarize ONLY what is explicitly stated. "
                  "Do not invent names, events, or outcomes. If unclear, say the transcript is unclear."
            )
        elif signal_words < 60:
            logger.warning(
                f"Moderate transcript signal (words={signal_words}); clamping to medium to stay faithful"
            )
            min_length, max_length = length_map.get("medium", (160, 260))
            context_prompt = (
                context_prompt
                + " Transcript is somewhat sparse/noisy. Summarize only what is explicitly stated. "
                  "If context is unclear, say so rather than guessing."
            )
        else:
            # Enough signal: honor user length choice but keep a safety ceiling.
            max_length = min(int(max_length), 520)

        logger.info(
            f"Combined transcript: chars={len(combined_text)}, words={len(combined_words)}, hash={_text_hash(combined_text)}"
        )

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

        if not summary_text or not summary_text.strip():
            return None
        if _looks_like_looping_hallucination(summary_text):
            stats = _repetition_stats(summary_text)
            logger.warning(
                "Model output looks repetitive/low-signal; using fallback | "
                f"words={stats.get('total')} uniq={stats.get('uniq')} top={stats.get('top_word')} "
                f"top_ratio={stats.get('top_ratio'):.2f} bg_top={stats.get('top_bigram_count')}"
            )
            return None
        logger.info(
            f"✓ Raw summary: chars={len(summary_text)}, boiler_hits={_boilerplate_hit_count(summary_text)}, "
            f"hash={_text_hash(summary_text)}"
        )
        return summary_text
    
    def _get_context_prompt(self, summary_type: str) -> str:
        """Return a prompt prefix based on summary_type."""
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
        """Generate summary text via Flan-T5."""
        text_length = self._normalize_text_length(text_length)
        word_targets = {
            "short": "100-250 words",
            "medium": "250-450 words",
            "long": "450-600 words",
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

        if logger.logger.isEnabledFor(10):
            logger.debug(
                "Prompt debug | "
                f"type={summary_type} length={text_length} prompt_chars={len(prompt)} prompt_hash={_text_hash(prompt)} | "
                f"context='{_preview(context_prompt, 180)}' transcript_preview='{_preview(text, 180)}'"
            )
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            length_penalty = {"short": 0.9, "medium": 1.0, "long": 1.10}.get(text_length, 1.0)
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
            if logger.logger.isEnabledFor(10):
                logger.debug(f"Model output preview: '{_preview(summary, 260)}'")

            cleaned = _clean_transcript_text(summary)
            if cleaned and cleaned.strip():
                summary = cleaned

            # Reject looping/low-signal summaries (often triggered by forcing long output on short input).
            if _looks_like_looping_hallucination(summary):
                return ""

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
        """Format the raw summary text."""
        text_length = self._normalize_text_length(text_length)
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
            result = self._to_plain(text, text_length=text_length)
            logger.info(f"  → Plain format applied")
            return _sanitize_profanity(result)

    def _ensure_min_lines(self, sentences: List[str], min_lines: int, max_lines: Optional[int] = None) -> List[str]:
        """Ensure at least min_lines by splitting long sentences."""
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
        """Convert summary to bullet points."""
        inline = _split_inline_bullets(text)
        if inline:
            sentences = _sanitize_lines([_sentence_case(x) for x in inline])
        else:
            sentences = _sanitize_lines(_split_sentences(text))

        target_lines = {
            "short": 8,
            "medium": 12,
            "long": 18,
        }.get(text_length, 10)

        bullets = sentences[:target_lines]
        if not bullets:
            return "• Unable to generate a clean bullet summary."

        logger.info(f"  → Bullets: requested={target_lines}, produced={len(bullets)}")

        formatted_bullets = "\n".join([f"• {_ensure_terminal_punct(b)}" for b in bullets])
        return formatted_bullets

    def _to_structured(self, text: str, summary_type: str, text_length: str) -> str:
        """Create a structured summary with simple sections."""
        tl = self._normalize_text_length(text_length)
        sentences = _sanitize_lines(_split_sentences(text))
        if not sentences:
            return "Unable to generate a clean structured summary."

        # Ensure terminal punctuation once up-front.
        sentences = [_ensure_terminal_punct(s) for s in sentences]

        if tl == "short":
            # Keep it compact but still structured.
            overview_sents = sentences[:3]
            key_points = sentences[3:8]

            overview = " ".join(overview_sents).strip()
            if not key_points and len(sentences) > 0:
                key_points = sentences[: min(3, len(sentences))]

            result = "Overview\n" + overview
            if key_points:
                result += "\n\nKey Points\n" + "\n".join([f"- {s}" for s in key_points])
            return result.strip()

        if tl == "medium":
            overview_sents = sentences[:4]
            key_points = sentences[4:14]

            overview = " ".join(overview_sents).strip()
            if not key_points and len(sentences) > 0:
                key_points = sentences[: min(6, len(sentences))]

            body = "\n".join([f"- {s}" for s in key_points])
            return f"Overview\n{overview}\n\nKey Points\n{body}".strip()

        # long
        overview_sents = sentences[:5]
        key_points = sentences[5:15]
        details = sentences[15:27]

        overview = " ".join(overview_sents).strip()

        logger.info(
            f"  → Structured sections: overview_sents={min(5, len(sentences))} key_points={len(key_points)} details={len(details)}"
        )

        result = f"Overview\n{overview}\n\n"
        if key_points:
            result += "Key Points\n" + "\n".join([f"- {s}" for s in key_points]) + "\n\n"
        if details:
            result += "Details\n" + "\n".join([f"- {s}" for s in details])
        return result.strip()
    
    def _to_plain(self, text: str, text_length: str = "medium") -> str:
        """Return plain paragraphs (multi-paragraph for medium/long)."""
        sentences = _sanitize_lines(_split_sentences(text))
        if not sentences:
            clean_text = " ".join(str(text).split())
            return _sentence_case(clean_text)

        tl = self._normalize_text_length(text_length)
        s = [_ensure_terminal_punct(x) for x in sentences]

        if tl == "short":
            out = " ".join(s[: min(5, len(s))]).strip()
            logger.info(f"  → Plain paragraphs: length=short paras=1 sents_used={min(5, len(s))}")
            return out

        if tl == "medium":
            if len(s) <= 6:
                out = " ".join(s).strip()
                logger.info(f"  → Plain paragraphs: length=medium paras=1 sents_used={len(s)}")
                return out
            p1 = " ".join(s[:4])
            p2 = " ".join(s[4:8])
            out = (p1 + "\n\n" + p2).strip()
            logger.info(f"  → Plain paragraphs: length=medium paras=2 sents_used={min(8, len(s))}")
            return out

        # long
        if len(s) <= 9:
            p1 = " ".join(s[:3])
            p2 = " ".join(s[3:6])
            p3 = " ".join(s[6:9]) if len(s) > 6 else ""
            out = (p1 + "\n\n" + p2 + ("\n\n" + p3 if p3 else "")).strip()
            logger.info(f"  → Plain paragraphs: length=long paras={2 + (1 if p3 else 0)} sents_used={min(9, len(s))}")
            return out

        p1 = " ".join(s[:4])
        p2 = " ".join(s[4:8])
        p3 = " ".join(s[8:12])
        p4 = " ".join(s[12:16]) if len(s) > 12 else ""
        out = (p1 + "\n\n" + p2 + "\n\n" + p3 + ("\n\n" + p4 if p4 else "")).strip()
        logger.info(
            f"  → Plain paragraphs: length=long paras={3 + (1 if p4 else 0)} sents_used={min(16, len(s))}"
        )
        return out
    
    def _generate_fallback_summary(self, summary_format: str, summary_type: str) -> str:
        """Fallback summary when transcripts are missing/low-signal."""
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
