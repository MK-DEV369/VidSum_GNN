import asyncio
import os
import torch
import shutil
import time
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from datetime import datetime, timedelta
import glob
from sqlalchemy.ext.asyncio import AsyncSession
from vidsum_gnn.db.client import AsyncSessionLocal
from vidsum_gnn.db.models import Video, Shot, Summary
from vidsum_gnn.processing.video import probe_video, transcode_video, merge_important_shots
from vidsum_gnn.processing.shot_detection import detect_shots, sample_frames_for_shots
from vidsum_gnn.processing.audio import extract_audio_segment
from vidsum_gnn.features.visual import VisualEncoder
from vidsum_gnn.features.audio import AudioEncoder
from vidsum_gnn.features.handcrafted import extract_handcrafted_features
from vidsum_gnn.inference.text_embedding import TextEmbedder
from vidsum_gnn.inference.transcription import TranscriptionService
from vidsum_gnn.graph.builder import GraphBuilder
from vidsum_gnn.graph.model import VidSumGNN
from vidsum_gnn.core.config import settings
from vidsum_gnn.utils.logging import get_logger
from vidsum_gnn.inference.service import get_inference_service
from vidsum_gnn.inference.gemini_fallback import get_gemini_summarizer
from vidsum_gnn.inference.summarization import _clean_transcript_text
from PIL import Image
import numpy as np

logger = get_logger(__name__)


def _safe_remove_path(path: str) -> None:
    """Remove a file or directory path if it exists; ignore errors."""
    if not path:
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _cleanup_video_artifacts(
    *,
    video_id: str,
    uploaded_file_path: str | None,
    canonical_path: str | None,
    keep_outputs: bool = True,
) -> None:
    """Cleanup per-video cache/intermediate files without breaking history by default."""
    # Upload file
    if getattr(settings, "CLEANUP_UPLOADS", True) and uploaded_file_path:
        _safe_remove_path(uploaded_file_path)

    # Processed dir (per-shot audio/keyframes/etc)
    if getattr(settings, "CLEANUP_PROCESSED", True):
        _safe_remove_path(os.path.join(settings.PROCESSED_DIR, video_id))
        # Defensive: remove any chunk processed dirs that may remain
        for p in glob.glob(os.path.join(settings.PROCESSED_DIR, f"{video_id}_chunk_*") ):
            _safe_remove_path(p)

    # Temp artifacts
    if getattr(settings, "CLEANUP_TEMP", True):
        # Per-video temp workspace (motion, etc)
        _safe_remove_path(os.path.join(settings.TEMP_DIR, video_id))
        # Any top-level temp files created for this video (canonical/chunks/concat/important_shots)
        for p in glob.glob(os.path.join(settings.TEMP_DIR, f"{video_id}_*")):
            _safe_remove_path(p)
        # Canonical file (sometimes uses original basename and may not match video_id prefix)
        if canonical_path and os.path.exists(canonical_path):
            _safe_remove_path(canonical_path)

    # Outputs are part of history (merged video + thumbnails + tts). Only delete if explicitly enabled.
    if not keep_outputs and getattr(settings, "CLEANUP_OUTPUTS", False):
        _safe_remove_path(os.path.join(settings.OUTPUT_DIR, "keyframes", video_id))
        _safe_remove_path(os.path.join(settings.OUTPUT_DIR, "merged", f"{video_id}.mp4"))
        _safe_remove_path(os.path.join(settings.OUTPUT_DIR, "tts", video_id))


def _evidence_bullet_from_transcript(text: str, shot_index: int) -> str:
    cleaned = _clean_transcript_text(text or "")
    if cleaned:
        garbage_words = {"commission", "times", "everyone", "importantly"}
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned.strip()) if p.strip()]
        # Prefer a sentence with at least ~3 content words.
        for p in parts[:4]:
            words = re.findall(r"[A-Za-z]{2,}", p.lower())
            if len(words) >= 3 and words[0] not in garbage_words:
                first = re.sub(r"\s+", " ", p).strip()
                if len(first) > 120:
                    first = first[:120].rstrip() + "…"
                return first
        # Fallback: accept a slightly shorter (but non-garbage) sentence.
        for p in parts[:4]:
            words = re.findall(r"[A-Za-z]{2,}", p.lower())
            if len(words) >= 2 and words[0] not in garbage_words:
                first = re.sub(r"\s+", " ", p).strip()
                if len(first) > 120:
                    first = first[:120].rstrip() + "…"
                return first
    return f"Shot {shot_index}"


def _first_meaningful_sentences(text: str, max_sentences: int = 2) -> list[str]:
    cleaned = _clean_transcript_text(text or "")
    if not cleaned:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned.strip()) if p.strip()]
    out: list[str] = []
    for p in parts[:8]:
        # Skip extremely short/noisy fragments
        words = re.findall(r"[A-Za-z]{2,}", p)
        if len(words) < 4:
            continue
        s = re.sub(r"\s+", " ", p).strip()
        if len(s) > 220:
            s = s[:220].rstrip() + "…"
        out.append(s)
        if len(out) >= max_sentences:
            break
    return out


def _evidence_heading_description(
    transcript_snippet: str,
    shot_index: int,
    signals: dict | None,
) -> tuple[str, str | None]:
    """Return (heading, description) for evidence.

    Heading should be short and scan-friendly; description can be longer.
    """
    heading = _evidence_bullet_from_transcript(transcript_snippet, shot_index)
    heading = re.sub(r"^[•\-\*\s]+", "", str(heading or "")).strip()

    sentences = _first_meaningful_sentences(transcript_snippet, max_sentences=2)
    description: str | None = None
    if sentences:
        # If the heading is basically the first sentence, use the next sentence as description.
        if heading and sentences and heading.rstrip(".!") == sentences[0].rstrip(".!"):
            description = sentences[1] if len(sentences) > 1 else None
        else:
            description = " ".join(sentences[:2]).strip() or None

    # If we still have no description, fall back to an interpretable signals summary.
    if not description:
        sig = signals or {}
        motion = sig.get("motion")
        audio_rms = sig.get("audio_rms")
        scene_change = sig.get("scene_change")
        transcript_density = sig.get("transcript_density")

        parts: list[str] = []
        if isinstance(motion, (int, float)) and not math.isnan(float(motion)):
            parts.append(f"motion={float(motion):.3f}")
        if isinstance(scene_change, (int, float)) and not math.isnan(float(scene_change)):
            parts.append(f"scene_change={float(scene_change):.3f}")
        if isinstance(audio_rms, (int, float)) and not math.isnan(float(audio_rms)):
            parts.append(f"audio_rms={float(audio_rms):.3f}")
        if isinstance(transcript_density, (int, float)) and not math.isnan(float(transcript_density)):
            parts.append(f"transcript_density={float(transcript_density):.3f}")

        if parts:
            description = "Selected due to " + ", ".join(parts[:3]) + "."

    if not heading:
        heading = f"Shot {shot_index}"
    if len(heading) > 140:
        heading = heading[:140].rstrip() + "…"
    return heading, description


def _evidence_justification(
    score: float | None,
    signals: dict | None,
    neighbors: list[dict] | None,
    normalized_signals: dict[str, float] | None = None,
) -> str:
    sig = signals or {}
    nrm = normalized_signals or {}

    # Prefer normalized signals when available, so "top reasons" is comparable.
    candidates: list[tuple[str, float]] = []
    for k in ("motion", "scene_change", "audio_rms", "transcript_density"):
        v = nrm.get(k)
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            candidates.append((k, float(v)))
    candidates = sorted(candidates, key=lambda kv: kv[1], reverse=True)
    top_reasons = [f"{k.replace('_', ' ')}" for k, _v in candidates[:2]]

    # Neighbor summary
    sem = [n for n in (neighbors or []) if n.get("edge_type") == "semantic"]
    tmp = [n for n in (neighbors or []) if n.get("edge_type") == "temporal"]
    best_sim = None
    try:
        best_sim = max([float(n.get("similarity") or 0.0) for n in sem]) if sem else None
    except Exception:
        best_sim = None
    closest_sec = None
    try:
        dists = [float(n.get("distance_sec")) for n in tmp if isinstance(n.get("distance_sec"), (int, float))]
        closest_sec = min([abs(d) for d in dists]) if dists else None
    except Exception:
        closest_sec = None

    parts: list[str] = []
    if isinstance(score, (int, float)) and not math.isnan(float(score)):
        parts.append(f"score={float(score):.4f}")
    if top_reasons:
        parts.append("top signals: " + ", ".join(top_reasons))
    if sem and best_sim is not None:
        parts.append(f"semantic neighbors: {len(sem)} (best sim {best_sim:.2f})")
    elif sem:
        parts.append(f"semantic neighbors: {len(sem)}")
    if tmp and closest_sec is not None:
        parts.append(f"temporal neighbors: {len(tmp)} (closest {int(round(closest_sec))}s)")
    elif tmp:
        parts.append(f"temporal neighbors: {len(tmp)}")

    return "; ".join(parts) if parts else "Selected based on importance score and graph signals."


def _normalize_metric(values: list[float | None]) -> list[float]:
    cleaned: list[float] = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if not cleaned:
        return [0.0 for _ in values]
    mn = float(min(cleaned))
    mx = float(max(cleaned))
    denom = (mx - mn) if (mx - mn) != 0 else 1.0
    out: list[float] = []
    for v in values:
        if not isinstance(v, (int, float)) or math.isnan(float(v)):
            out.append(0.0)
        else:
            out.append(max(0.0, min(1.0, (float(v) - mn) / denom)))
    return out


def _adjust_scores_for_summary_type(
    summary_type: str,
    base_scores: list[float],
    transcripts: list[str],
    shots_times: list[tuple[float, float]],
    audio_paths: list[str],
    keyframe_paths: list[str],
) -> list[float]:
    """Adjust GNN scores using lightweight proxies so summary_type influences shot selection.

    Note: This is heuristic. True audio/visual understanding would require modality-specific models.
    """
    st = str(summary_type or "balanced")
    if st == "balanced":
        return [float(s) for s in base_scores]

    n = len(base_scores)
    durations: list[float | None] = []
    densities: list[float | None] = []
    audio_rms: list[float | None] = []
    scene_change: list[float | None] = []

    for i in range(n):
        try:
            start, end = shots_times[i] if i < len(shots_times) else (0.0, 0.0)
            d = float(end) - float(start)
            durations.append(max(0.0, d))
        except Exception:
            durations.append(None)

        try:
            wc = _safe_word_count(transcripts[i] if i < len(transcripts) else "")
            d2 = durations[-1] if durations else None
            if isinstance(d2, (int, float)) and d2 > 0:
                densities.append(float(wc) / max(0.25, float(d2)))
            else:
                densities.append(float(wc))
        except Exception:
            densities.append(None)

        try:
            ap = audio_paths[i] if i < len(audio_paths) else None
            audio_rms.append(_audio_rms_energy(ap) if ap else None)
        except Exception:
            audio_rms.append(None)

        try:
            if i <= 0:
                scene_change.append(None)
            else:
                prev_kf = keyframe_paths[i - 1] if i - 1 < len(keyframe_paths) else None
                cur_kf = keyframe_paths[i] if i < len(keyframe_paths) else None
                if prev_kf and cur_kf:
                    scene_change.append(_histogram_delta(prev_kf, cur_kf))
                else:
                    scene_change.append(None)
        except Exception:
            scene_change.append(None)

    dur_n = _normalize_metric(durations)
    dens_n = _normalize_metric(densities)
    audio_n = _normalize_metric(audio_rms)
    scene_n = _normalize_metric(scene_change)

    adjusted: list[float] = []
    for i in range(n):
        base = float(base_scores[i])
        if st == "audio_priority":
            s = (0.70 * base) + (0.20 * audio_n[i]) + (0.10 * dens_n[i])
        elif st == "visual_priority":
            s = (0.70 * base) + (0.20 * scene_n[i]) + (0.10 * dur_n[i])
        elif st == "highlights":
            s = (0.60 * base) + (0.20 * audio_n[i]) + (0.20 * scene_n[i])
        else:
            s = base
        adjusted.append(float(s))

    return adjusted

# Batch processing constants
CHUNK_DURATION = int(os.getenv("LONG_VIDEO_CHUNK_DURATION", "900"))  # seconds
MAX_VIDEO_DURATION = int(os.getenv("LONG_VIDEO_MAX_DURATION", "3600"))  # seconds

# Long-video tuning defaults (safe for 8GB GPU like RTX 3070)
LONG_SHOT_THRESHOLD = float(os.getenv("LONG_VIDEO_SHOT_THRESHOLD", "0.45"))
LONG_MIN_SHOT_DURATION = float(os.getenv("LONG_VIDEO_MIN_SHOT_DURATION", "1.5"))
LONG_MAX_SHOTS_PER_CHUNK = int(os.getenv("LONG_VIDEO_MAX_SHOTS_PER_CHUNK", "240"))
LONG_TOPK_RATIO = float(os.getenv("LONG_VIDEO_TOPK_RATIO", "0.03"))
LONG_TOPK_CAP = int(os.getenv("LONG_VIDEO_TOPK_CAP", "80"))


def _time_segment_shots(duration_sec: float, segments: int) -> list[tuple[float, float]]:
    """Fallback segmentation when shot detection finds too few cuts.

    Creates approximately-uniform segments across the full duration.
    """
    try:
        duration = float(max(0.0, duration_sec))
        if duration <= 0.0:
            return []
        segs = int(max(1, segments))
        seg_len = duration / float(segs)
        out: list[tuple[float, float]] = []
        t = 0.0
        for _ in range(segs):
            start = t
            end = min(duration, start + seg_len)
            if end - start >= 0.5:
                out.append((float(start), float(end)))
            t = end
            if t >= duration:
                break
        # Ensure last segment ends at duration
        if out and out[-1][1] < duration:
            out[-1] = (out[-1][0], float(duration))
        return out
    except Exception:
        return []


def _merge_shots_to_cap(shots: list[tuple[float, float]], max_shots: int) -> list[tuple[float, float]]:
    """Merge consecutive shots to cap total count while preserving order."""
    if max_shots <= 0 or len(shots) <= max_shots:
        return shots
    factor = int(math.ceil(len(shots) / max_shots))
    merged: list[tuple[float, float]] = []
    for i in range(0, len(shots), factor):
        block = shots[i:i + factor]
        if not block:
            continue
        merged.append((float(block[0][0]), float(block[-1][1])))
    return merged


def _safe_word_count(text: str) -> int:
    if not text:
        return 0
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _normalize_01(value: float | None, vmin: float, vmax: float) -> float | None:
    if value is None:
        return None
    if vmax <= vmin:
        return 0.0
    x = (value - vmin) / (vmax - vmin)
    return float(max(0.0, min(1.0, x)))


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into H:MM:SS or M:SS string."""
    try:
        total = max(0, int(round(seconds)))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"
    except Exception:
        return "--:--"


def _histogram_delta(img_a_path: str | None, img_b_path: str | None) -> float | None:
    """Fast scene-change proxy based on grayscale histogram L1 distance."""
    if not img_a_path or not img_b_path:
        return None
    try:
        with Image.open(img_a_path).convert("L") as a, Image.open(img_b_path).convert("L") as b:
            ha = np.array(a.histogram(), dtype=np.float32)
            hb = np.array(b.histogram(), dtype=np.float32)
        ha /= max(1.0, float(ha.sum()))
        hb /= max(1.0, float(hb.sum()))
        return float(np.abs(ha - hb).sum() / 2.0)  # [0,1]
    except Exception:
        return None


async def _extract_frame_to_path(video_path: str, timestamp: float, output_path: str) -> bool:
    cmd = [
        "ffmpeg",
        "-ss", str(max(0.0, float(timestamp))),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y",
        output_path,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, _ = await proc.communicate()
        return proc.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


async def _motion_estimate(video_path: str, start_sec: float, end_sec: float, temp_dir: str) -> float | None:
    """Cheap motion proxy: mean abs pixel diff between two frames inside the shot."""
    dur = max(0.001, float(end_sec - start_sec))
    t1 = float(start_sec + 0.2 * dur)
    t2 = float(end_sec - 0.2 * dur)
    if t2 <= t1:
        t2 = float(start_sec + 0.8 * dur)

    os.makedirs(temp_dir, exist_ok=True)
    p1 = os.path.join(temp_dir, "m1.jpg")
    p2 = os.path.join(temp_dir, "m2.jpg")
    ok1 = await _extract_frame_to_path(video_path, t1, p1)
    ok2 = await _extract_frame_to_path(video_path, t2, p2)
    if not (ok1 and ok2):
        return None
    try:
        with Image.open(p1).convert("L") as a, Image.open(p2).convert("L") as b:
            aa = np.asarray(a, dtype=np.float32)
            bb = np.asarray(b, dtype=np.float32)
        if aa.shape != bb.shape:
            return None
        diff = np.abs(aa - bb)
        return float(diff.mean() / 255.0)
    except Exception:
        return None
    finally:
        for p in (p1, p2):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def _audio_rms_energy(audio_path: str | None) -> float | None:
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        import torchaudio
        waveform, _sr = torchaudio.load(audio_path)
        if waveform.numel() == 0:
            return None
        # waveform: (channels, samples)
        x = waveform.float()
        rms = torch.sqrt(torch.mean(x * x)).item()
        return float(rms)
    except Exception:
        return None


def _graph_neighbors(graph_data, shot_index: int, max_neighbors: int = 8) -> list[dict]:
    """Return up to max_neighbors neighbor descriptors using edge_attr.

    edge_attr format: [is_temporal, distance_norm, sim, audio_corr]
    """
    try:
        edge_index = graph_data.edge_index
        edge_attr = getattr(graph_data, "edge_attr", None)
        if edge_index is None:
            return []
        src = edge_index[0].detach().cpu().numpy()
        dst = edge_index[1].detach().cpu().numpy()
        attr = edge_attr.detach().cpu().numpy() if edge_attr is not None else None

        rows: list[dict] = []
        for e, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
            if int(s) != int(shot_index):
                continue
            if attr is not None and e < attr.shape[0]:
                is_temporal = float(attr[e][0]) >= 0.5
                distance_sec = float(attr[e][1]) * 60.0
                sim = float(attr[e][2])
            else:
                is_temporal = False
                distance_sec = 0.0
                sim = 0.0

            rows.append({
                "neighbor_index": int(d),
                "edge_type": "temporal" if is_temporal else "semantic",
                "similarity": sim if not is_temporal else None,
                "distance_sec": distance_sec if is_temporal else None,
            })

        # Prefer semantic by similarity, then temporal by closest distance
        semantic = sorted([r for r in rows if r["edge_type"] == "semantic"], key=lambda r: (r["similarity"] or 0.0), reverse=True)
        temporal = sorted([r for r in rows if r["edge_type"] == "temporal"], key=lambda r: abs(r["distance_sec"] or 0.0))
        merged: list[dict] = []
        seen: set[int] = set()
        for r in semantic + temporal:
            ni = int(r["neighbor_index"])
            if ni in seen:
                continue
            seen.add(ni)
            merged.append(r)
            if len(merged) >= max_neighbors:
                break
        return merged
    except Exception:
        return []


_STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","to","of","in","on","for","with",
    "is","are","was","were","be","been","being","as","at","by","from","this","that","these","those","it",
    "we","you","they","i","he","she","them","us","our","your","their","my","me","his","her","so","not",
    "can","could","should","would","will","just","also","about","into","over","than","too","very","up","down",
    # Common speech / filler that tends to dominate chapter keyword titles
    "what","why","how","who","where","when","which","exactly","actually","basically","literally","really",
    "okay","ok","right","like","yeah","yes","no","uh","um","hmm","well","maybe","kind","kinda","sort","sorta",
    "thing","things","stuff","people","someone","something",
    "doing","does","did","done","make","makes","made","adding","addition","add",
    # Noisy ASR artifacts that frequently dominate chapter titles
    "commission","times","everyone","importantly","start",
}


def _chapter_title(texts: list[str], index: int) -> str:
    blob = " ".join([t for t in texts if t]).strip()
    if not blob:
        return f"Chapter {index + 1}"
    words = [w.lower() for w in re.findall(r"[a-zA-Z]{3,}", blob)]
    words = [w for w in words if w not in _STOPWORDS]

    if not words:
        return f"Chapter {index + 1}"

    # Prefer slightly more specific keywords: skip very low-signal repeats.
    counts = Counter(words)

    # If the transcript is essentially one repeated word, treat as low-signal and fallback.
    try:
        most_common = counts.most_common(1)
        if most_common:
            top_word, top_count = most_common[0]
            if len(counts) <= 2 and (top_count / max(1, len(words))) >= 0.75:
                return f"Chapter {index + 1}"
    except Exception:
        pass

    candidates: list[str] = []
    for w, _c in counts.most_common(12):
        if w in _STOPWORDS:
            continue
        # Avoid overly generic stems that still slip through.
        if w in {"video", "audio", "sound"}:
            continue
        candidates.append(w)
        if len(candidates) >= 3:
            break

    if len(candidates) < 2:
        # Fall back to a short snippet of content words.
        snippet_words_raw = [w for w in words if w not in _STOPWORDS][:12]
        seen: set[str] = set()
        snippet_words: list[str] = []
        for w in snippet_words_raw:
            if w in seen:
                continue
            seen.add(w)
            snippet_words.append(w)
            if len(snippet_words) >= 6:
                break
        snippet = " ".join([w.capitalize() for w in snippet_words]).strip()
        return snippet or f"Chapter {index + 1}"

    title = " · ".join([w.capitalize() for w in candidates])
    return title or f"Chapter {index + 1}"


def _chapter_description(texts: list[str], index: int, title: str) -> str:
    blob = " ".join([t for t in texts if t]).strip()
    if blob:
        sentences = _first_meaningful_sentences(blob, max_sentences=2)
        if sentences:
            desc = " ".join(sentences[:2]).strip()
            # Avoid repeating the title keywords verbatim as the whole description.
            if desc and desc.lower() != (title or "").lower():
                return desc
    return f"Chapter {index + 1} segment in the merged summary."


def _chapter_keywords(texts: list[str], max_keywords: int = 5) -> list[str]:
    blob = " ".join([t for t in texts if t]).strip()
    if not blob:
        return []
    words = [w.lower() for w in re.findall(r"[a-zA-Z]{3,}", blob)]
    words = [w for w in words if w not in _STOPWORDS]
    if not words:
        return []
    counts = Counter(words)
    out: list[str] = []
    for w, _c in counts.most_common(20):
        if w in _STOPWORDS or w in {"video", "audio", "sound"}:
            continue
        out.append(w)
        if len(out) >= max_keywords:
            break
    return out


def _build_chapters_from_manifest(
    merged_manifest: list[dict],
    transcriptions: list[str],
    min_chapters: int = 5,
    max_chapters: int = 12,
) -> list[dict]:
    """Group merged segments into YouTube-style chapters."""
    segs = [m for m in (merged_manifest or []) if isinstance(m, dict) and m.get("merged_start") is not None]
    if not segs:
        return []

    segs = sorted(segs, key=lambda m: float(m.get("merged_start") or 0.0))
    merged_end = float(max([float(m.get("merged_end") or 0.0) for m in segs] + [0.0]))
    # Choose chapter count based on duration (~45s per chapter) and bounds
    target = int(round(max(1.0, merged_end / 45.0)))
    k = max(min_chapters, min(max_chapters, target))
    k = min(k, len(segs))

    # Partition into k groups by cumulative duration
    total_dur = sum([max(0.0, float(m.get("merged_end") or 0.0) - float(m.get("merged_start") or 0.0)) for m in segs])
    per = total_dur / max(1, k)
    chapters: list[dict] = []

    cur: list[dict] = []
    cur_d = 0.0
    for seg in segs:
        seg_d = max(0.0, float(seg.get("merged_end") or 0.0) - float(seg.get("merged_start") or 0.0))
        if cur and len(chapters) < k - 1 and (cur_d + seg_d) > per:
            chapters.append({"segments": cur})
            cur = []
            cur_d = 0.0
        cur.append(seg)
        cur_d += seg_d
    if cur:
        chapters.append({"segments": cur})

    out: list[dict] = []
    for idx, ch in enumerate(chapters):
        s0 = ch["segments"][0]
        s1 = ch["segments"][-1]
        shot_indices = [int(s.get("shot_index")) for s in ch["segments"] if s.get("shot_index") is not None]
        texts = [transcriptions[i] for i in shot_indices if 0 <= i < len(transcriptions)]
        title = _chapter_title(texts, idx)
        out.append({
            "index": idx,
            "title": title,
            "description": _chapter_description(texts, idx, title),
            "keywords": _chapter_keywords(texts, max_keywords=5),
            "merged_start": float(s0.get("merged_start") or 0.0),
            "merged_end": float(s1.get("merged_end") or 0.0),
            "shot_indices": shot_indices,
        })
    return out

async def _send_log(video_id: str, message: str, level: str = "INFO", stage: str | None = None, progress: int | None = None):
    from vidsum_gnn.api.main import manager
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level,
        "message": message,
    }
    if stage is not None:
        payload["stage"] = stage
    if progress is not None:
        payload["progress"] = progress

    try:
        await manager.send_log(video_id, payload)
    except Exception as e:
        logger.error(f"Failed to broadcast log for {video_id}: {e}")


async def estimate_processing_time(video_duration: float, is_batch: bool = False) -> str:
    """Estimate time for video processing based on duration"""
    # Rough estimates per stage:
    # Transcoding: 0.5x speed
    # Shot detection: 0.1x speed  
    # Feature extraction: 0.3x speed
    # GNN inference: 0.2x speed per chunk
    # Total: ~1.1x per regular chunk, 2x for batch processing
    
    multiplier = 2.0 if is_batch else 1.1
    estimated_seconds = video_duration * multiplier
    
    hours = int(estimated_seconds // 3600)
    minutes = int((estimated_seconds % 3600) // 60)
    seconds = int(estimated_seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


async def process_video_chunk(
    db: AsyncSession,
    video_id: str,
    canonical_path: str,
    chunk_start: float,
    chunk_end: float,
    chunk_index: int,
    total_chunks: int,
    base_progress: int,
    progress_per_chunk: int,
    video_duration_seconds: float
):
    """Process a single chunk of video and return (shots, scores, neighbors).

    For long videos we intentionally skip per-shot Whisper transcription inside each
    chunk (it is the dominant cost). We keep feature dims consistent by using
    zero text embeddings in chunk mode.
    """
    logger.info(f"Processing chunk {chunk_index + 1}/{total_chunks} ({chunk_start:.0f}s - {chunk_end:.0f}s)")
    await _send_log(
        video_id, 
        f"Processing chunk {chunk_index + 1}/{total_chunks} ({chunk_start:.0f}s - {chunk_end:.0f}s)",
        stage="batch_processing",
        progress=base_progress + (chunk_index * progress_per_chunk)
    )
    
    # Create temporary chunk video file
    chunk_path = os.path.join(settings.TEMP_DIR, f"{video_id}_chunk_{chunk_index:03d}.mp4")
    extract_cmd = [
        "ffmpeg",
        "-i", canonical_path,
        "-ss", str(chunk_start),
        "-to", str(chunk_end),
        "-c", "copy",
        "-y",
        chunk_path
    ]
    
    process = await asyncio.create_subprocess_exec(
        *extract_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
    
    if process.returncode != 0:
        logger.warning(f"Stream-copy chunk extract failed (chunk {chunk_index}): {stderr.decode()[:200]}")

        # Fallback: re-encode the chunk (slower but robust across GOP boundaries)
        extract_cmd = [
            "ffmpeg",
            "-ss", str(chunk_start),
            "-to", str(chunk_end),
            "-i", canonical_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            "-y",
            chunk_path,
        ]
        process = await asyncio.create_subprocess_exec(
            *extract_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout2, stderr2 = await asyncio.wait_for(process.communicate(), timeout=900)
        if process.returncode != 0:
            logger.error(f"Failed to extract chunk {chunk_index} with re-encode: {stderr2.decode()[:200]}")
            return None, None, None
    
    # Run feature extraction and GNN on chunk
    try:
        shots_times = await detect_shots(chunk_path, threshold=LONG_SHOT_THRESHOLD)

        # Filter and cap shots to keep chunk processing bounded.
        shots_times = [(s, e) for s, e in shots_times if (e - s) >= LONG_MIN_SHOT_DURATION]
        shots_times = _merge_shots_to_cap(shots_times, LONG_MAX_SHOTS_PER_CHUNK)

        keyframe_paths, frames_per_shot = await sample_frames_for_shots(
            chunk_path,
            shots_times,
            f"{video_id}_chunk_{chunk_index}",
            frames_per_shot=3,
        )
        
        audio_dir = os.path.join(settings.PROCESSED_DIR, f"{video_id}_chunk_{chunk_index}", "audio")
        os.makedirs(audio_dir, exist_ok=True)
        audio_paths = []
        
        for i, (start, end) in enumerate(shots_times):
            path = os.path.join(audio_dir, f"shot_{i:04d}.mp3")
            await extract_audio_segment(chunk_path, start, end, path)
            audio_paths.append(path)
        
        vis_encoder = VisualEncoder()
        aud_encoder = AudioEncoder()
        
        frame_feats = vis_encoder.encode(keyframe_paths)  # (sum frames, 768)
        # Aggregate frame embeddings back to shot-level (mean pooling)
        shot_embs = []
        offset = 0
        for n in frames_per_shot:
            if n <= 0:
                shot_embs.append(torch.zeros(768))
                continue
            slice_feats = frame_feats[offset : offset + n]
            offset += n
            if slice_feats.numel() == 0:
                shot_embs.append(torch.zeros(768))
            else:
                shot_embs.append(slice_feats.mean(dim=0))
        vis_feats = torch.stack(shot_embs) if shot_embs else torch.empty(0, 768)

        aud_feats = aud_encoder.encode(audio_paths)     # (N, 768)
        
        # Clear CUDA memory after feature extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Extract handcrafted features (duration, position, etc.)
        handcrafted_feats = extract_handcrafted_features(
            shots_times=shots_times,
            video_duration=video_duration_seconds
        )  # (N, 14)
        
        # Long-video optimization: skip Whisper + text embedding here.
        # Keep the feature vector shape consistent by using zeros for text dims.
        text_feats = torch.zeros(len(shots_times), 384, dtype=torch.float32)
        
        # Concatenate all features: 768 + 768 + 14 + 384 = 1934
        features = torch.cat([
            vis_feats,
            aud_feats,
            handcrafted_feats,
            text_feats
        ], dim=1)
        
        logger.info(f"Feature composition: Visual={vis_feats.shape[1]}, Audio={aud_feats.shape[1]}, Handcrafted={handcrafted_feats.shape[1]}, Text={text_feats.shape[1]}, Total={features.shape[1]}")
        
        shots_data = [
            {
                "shot_id": f"{video_id}_chunk{chunk_index}_{i:04d}",
                "video_id": video_id,
                "start_sec": chunk_start + start,
                "end_sec": chunk_start + end,
                "duration_sec": end - start
            }
            for i, (start, end) in enumerate(shots_times)
        ]
        
        builder = GraphBuilder()
        graph_data = builder.build_graph(shots_data, features)
        
        inference_service = get_inference_service()
        gnn_scores = inference_service.predict_importance_scores(
            graph_data.x,
            graph_data.edge_index
        )
        
        # Convert to plain list
        scores = gnn_scores.reshape(-1).tolist() if hasattr(gnn_scores, "reshape") else list(gnn_scores)

        # Precompute neighbor lists for explainability (cheap)
        neighbors_by_local_index = [
            _graph_neighbors(graph_data, i, max_neighbors=8)
            for i in range(len(shots_times))
        ]
        
        adjusted_shots = [
            (chunk_start + start, chunk_start + end)
            for start, end in shots_times
        ]
        
        # Cleanup chunk
        try:
            os.remove(chunk_path)
            shutil.rmtree(os.path.dirname(audio_dir), ignore_errors=True)
        except:
            pass
        
        return adjusted_shots, scores, neighbors_by_local_index
        
    except Exception as e:
        logger.error(f"Chunk {chunk_index} processing failed: {e}")
        return None, None, None


async def process_video_task(video_id: str, config: dict):
    """
    Main pipeline orchestration.
    """
    async with AsyncSessionLocal() as db:
        try:
            # Processing time tracking (admin logging)
            processing_started_at = datetime.utcnow()
            processing_started_mono = time.monotonic()

            # 1. Update status
            video = await db.get(Video, video_id)
            if not video:
                return
            video.status = "preprocessing"
            await db.commit()
            await _send_log(video_id, "Preprocessing started", stage="preprocessing", progress=20)
            
            # 2. Preprocessing
            input_path = os.path.join(settings.UPLOAD_DIR, video.filename)
            logger.info(f"Input path: {input_path}, exists: {os.path.exists(input_path)}")
            await _send_log(video_id, "Starting transcoding", stage="preprocessing", progress=25)
            canonical_path = await transcode_video(input_path)
            logger.info(f"Transcoding successful, output: {canonical_path}")
            await _send_log(video_id, "Transcoding complete", stage="preprocessing", progress=30)
            
            # Probe video for metadata
            video_metadata = await probe_video(canonical_path)
            video_duration_seconds = video_metadata['duration']
            logger.info(f"Video duration: {video_duration_seconds:.1f}s")
            await _send_log(video_id, f"Video duration: {video_duration_seconds:.1f}s", stage="preprocessing", progress=32)

            # LONG VIDEO PATH (movies): chunked processing
            if video_duration_seconds > MAX_VIDEO_DURATION:
                await _send_log(
                    video_id,
                    f"Long video detected ({video_duration_seconds/60:.1f} min). Using chunked processing for stability.",
                    stage="batch_processing",
                    progress=33,
                )

                total_chunks = int(math.ceil(video_duration_seconds / float(CHUNK_DURATION)))
                base_progress = 35
                progress_per_chunk = max(1, int(40 / max(1, total_chunks)))

                all_shots_times: list[tuple[float, float]] = []
                all_scores: list[float] = []
                all_neighbors: list[list[dict]] = []

                for chunk_index in range(total_chunks):
                    chunk_start = float(chunk_index * CHUNK_DURATION)
                    chunk_end = float(min(video_duration_seconds, (chunk_index + 1) * CHUNK_DURATION))
                    shots_part, scores_part, neighbors_part = await process_video_chunk(
                        db=db,
                        video_id=video_id,
                        canonical_path=canonical_path,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                        base_progress=base_progress,
                        progress_per_chunk=progress_per_chunk,
                        video_duration_seconds=video_duration_seconds,
                    )
                    if not shots_part or not scores_part:
                        continue

                    offset = len(all_shots_times)
                    all_shots_times.extend(shots_part)
                    all_scores.extend([float(s) for s in scores_part])

                    # Adjust neighbor indices into global index space
                    if neighbors_part:
                        for i, nlist in enumerate(neighbors_part):
                            adj: list[dict] = []
                            for n in (nlist or []):
                                nn = dict(n)
                                if "neighbor_index" in nn:
                                    nn["neighbor_index"] = int(nn["neighbor_index"]) + offset
                                adj.append(nn)
                            all_neighbors.append(adj)
                    else:
                        all_neighbors.extend([[] for _ in range(len(shots_part))])

                if not all_shots_times:
                    raise Exception("Chunked processing produced no shots")

                await _send_log(
                    video_id,
                    f"Chunked processing complete: {len(all_shots_times)} shots scored.",
                    stage="gnn_inference",
                    progress=75,
                )

                # Persist shots with importance scores (keyframes added later only for evidence shots)
                for i, ((start, end), score) in enumerate(zip(all_shots_times, all_scores)):
                    shot = Shot(
                        shot_id=f"{video_id}_{i:04d}",
                        video_id=video_id,
                        start_sec=float(start),
                        end_sec=float(end),
                        duration_sec=float(end - start),
                        importance_score=float(score),
                    )
                    db.add(shot)
                await db.commit()

                inference_service = get_inference_service()
                summary_type = config.get("summary_type", "balanced")
                text_length = config.get("text_length", "medium")
                summary_format = config.get("summary_format", "bullet")

                # Select top indices for transcription + explainability (cap absolute count)
                scores_arr = np.array(all_scores, dtype=float)
                n = int(scores_arr.size)
                top_k = max(1, int(math.ceil(max(0.0, min(1.0, LONG_TOPK_RATIO)) * n)))
                top_k = int(min(top_k, LONG_TOPK_CAP, n))
                top_indices = scores_arr.argsort()[::-1][:top_k].tolist() if n > 0 else []
                top_indices_sorted = sorted([int(i) for i in top_indices])

                await _send_log(
                    video_id,
                    f"Transcribing top-{top_k} shots only (movie optimization)",
                    stage="summarization",
                    progress=82,
                )

                # Transcribe only top shots for summarization/evidence
                transcriber = inference_service.manager.get_whisper()
                transcripts: list[str] = ["" for _ in range(n)]

                audio_tmp_dir = os.path.join(settings.TEMP_DIR, video_id, "top_audio")
                os.makedirs(audio_tmp_dir, exist_ok=True)
                for idx in top_indices_sorted:
                    start_sec, end_sec = all_shots_times[idx]
                    audio_path = os.path.join(audio_tmp_dir, f"shot_{idx:04d}.mp3")
                    try:
                        await extract_audio_segment(canonical_path, float(start_sec), float(end_sec), audio_path)
                        t = transcriber.transcribe(Path(audio_path), cache_dir=None)
                        transcripts[idx] = _clean_transcript_text((t or "").strip())
                    except Exception as e:
                        logger.warning(f"Top-shot transcription failed (idx={idx}): {e}")

                # Reweight scores for summary_type using proxies we can compute for the top shots.
                # (We don't have full-video per-shot audio/motion at this stage.)
                scores_for_summary = [float(s) for s in all_scores]
                if summary_type in ("audio_priority", "visual_priority", "highlights") and top_indices_sorted:
                    top_dur: list[float | None] = []
                    top_dens: list[float | None] = []
                    top_audio: list[float | None] = []
                    for idx in top_indices_sorted:
                        try:
                            start_sec, end_sec = all_shots_times[idx]
                            d = float(end_sec) - float(start_sec)
                            d = max(0.0, d)
                        except Exception:
                            d = 0.0
                        top_dur.append(d)

                        try:
                            wc = _safe_word_count(transcripts[idx] if idx < len(transcripts) else "")
                            top_dens.append(float(wc) / max(0.25, float(d)) if d > 0 else float(wc))
                        except Exception:
                            top_dens.append(None)

                        try:
                            ap = os.path.join(audio_tmp_dir, f"shot_{idx:04d}.mp3")
                            top_audio.append(_audio_rms_energy(ap))
                        except Exception:
                            top_audio.append(None)

                    dur_n = _normalize_metric(top_dur)
                    dens_n = _normalize_metric(top_dens)
                    audio_n = _normalize_metric(top_audio)

                    for j, idx in enumerate(top_indices_sorted):
                        base = float(scores_for_summary[idx])
                        if summary_type == "audio_priority":
                            scores_for_summary[idx] = (0.70 * base) + (0.20 * audio_n[j]) + (0.10 * dens_n[j])
                        elif summary_type == "visual_priority":
                            # Best-effort proxy here: longer shots often reflect visual continuity.
                            scores_for_summary[idx] = (0.80 * base) + (0.20 * dur_n[j])
                        elif summary_type == "highlights":
                            scores_for_summary[idx] = (0.70 * base) + (0.30 * audio_n[j])

                summarizer = inference_service.manager.get_summarizer()
                try:
                    nonempty = sum(1 for t in transcripts if t and t.strip())
                    total_words = sum(_safe_word_count(t) for t in transcripts if t and t.strip())
                    logger.info(f"Transcript stats: nonempty={nonempty}/{len(transcripts)}, total_words={total_words}")
                    if total_words < 25:
                        logger.warning(
                            "Transcript signal is sparse/noisy; summary may be shorter or rely more on visual cues"
                        )
                except Exception:
                    # Never let diagnostics break the pipeline.
                    pass
                all_formats = summarizer.summarize_all_formats(
                    transcripts=transcripts,
                    gnn_scores=scores_for_summary,
                    summary_type=summary_type,
                    text_length=text_length,
                    top_k=top_k,
                    video_path=canonical_path,
                    formats=[summary_format],
                )

                # Persist keyframes only for evidence shots
                keyframes_out_dir = os.path.join(settings.OUTPUT_DIR, "keyframes", video_id)
                os.makedirs(keyframes_out_dir, exist_ok=True)

                evidence_items: list[dict] = []
                for j, idx in enumerate(top_indices_sorted):
                    start_sec, end_sec = all_shots_times[idx]
                    center = float(start_sec + (end_sec - start_sec) / 2.0)
                    kf_path = os.path.join(keyframes_out_dir, f"shot_{idx:04d}.jpg")
                    await _extract_frame_to_path(canonical_path, center, kf_path)

                    # Update shot record with keyframe path
                    shot_obj = await db.get(Shot, f"{video_id}_{idx:04d}")
                    if shot_obj is not None:
                        shot_obj.keyframe_path = kf_path

                    transcript_snippet = _clean_transcript_text(transcripts[idx] if idx < len(transcripts) else "")
                    transcript_snippet = (transcript_snippet or "").strip()
                    if len(transcript_snippet) > 260:
                        transcript_snippet = transcript_snippet[:260] + "…"

                    duration = float(end_sec - start_sec)
                    wc = _safe_word_count(transcripts[idx] if idx < len(transcripts) else "")
                    transcript_density = (wc / max(0.25, duration)) if duration > 0 else float(wc)
                    audio_rms = _audio_rms_energy(os.path.join(audio_tmp_dir, f"shot_{idx:04d}.mp3"))
                    prev_kf = os.path.join(keyframes_out_dir, f"shot_{idx-1:04d}.jpg") if idx > 0 else None
                    scene_change = _histogram_delta(prev_kf if prev_kf and os.path.exists(prev_kf) else None, kf_path)
                    motion_tmp = os.path.join(settings.TEMP_DIR, video_id, "motion", f"shot_{idx:04d}")
                    motion = await _motion_estimate(canonical_path, float(start_sec), float(end_sec), motion_tmp)
                    neighbors = all_neighbors[idx] if idx < len(all_neighbors) else []

                    # Build user-facing heading/description from transcript+signals.
                    signals_obj = {
                        "motion": motion,
                        "audio_rms": audio_rms,
                        "scene_change": scene_change,
                        "transcript_density": float(transcript_density),
                        "duration_sec": float(duration),
                    }
                    heading, description = _evidence_heading_description(
                        transcript_snippet=transcript_snippet,
                        shot_index=int(idx),
                        signals=signals_obj,
                    )

                    evidence_items.append({
                        "index": j,
                        "shot_index": int(idx),
                        "shot_id": f"{video_id}_{idx:04d}",
                        "orig_start": float(start_sec),
                        "orig_end": float(end_sec),
                        "score": float(all_scores[idx]) if idx < len(all_scores) else None,
                        # Backward-compatible: keep `bullet` as the primary heading.
                        "bullet": heading,
                        "description": description,
                        "transcript_snippet": transcript_snippet,
                        "thumbnail_path": kf_path,
                        "signals": {
                            **signals_obj,
                        },
                        "neighbors": neighbors,
                    })

                # Add a compact justification string per evidence item (computed after we have the full set,
                # so we can normalize signals within the evidence list).
                try:
                    keys = ["motion", "scene_change", "audio_rms", "transcript_density"]
                    mins: dict[str, float] = {}
                    maxs: dict[str, float] = {}
                    for k in keys:
                        vals: list[float] = []
                        for it in evidence_items:
                            v = (it.get("signals") or {}).get(k)
                            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                                vals.append(float(v))
                        if vals:
                            mins[k] = min(vals)
                            maxs[k] = max(vals)

                    for it in evidence_items:
                        sig = it.get("signals") or {}
                        normalized: dict[str, float] = {}
                        for k in keys:
                            v = sig.get(k)
                            if not isinstance(v, (int, float)) or math.isnan(float(v)):
                                continue
                            mn = mins.get(k)
                            mx = maxs.get(k)
                            if mn is None or mx is None:
                                continue
                            denom = (mx - mn) if (mx - mn) != 0 else 1.0
                            normalized[k] = max(0.0, min(1.0, (float(v) - float(mn)) / float(denom)))
                        it["justification"] = _evidence_justification(
                            score=it.get("score"),
                            signals=sig,
                            neighbors=it.get("neighbors") or [],
                            normalized_signals=normalized,
                        )
                except Exception:
                    for it in evidence_items:
                        if "justification" not in it:
                            it["justification"] = _evidence_justification(
                                score=it.get("score"),
                                signals=it.get("signals") or {},
                                neighbors=it.get("neighbors") or [],
                                normalized_signals=None,
                            )

                await db.commit()

                # Create merged video + manifest
                await _send_log(video_id, "Creating merged video of important shots", stage="video_merge", progress=92)
                try:
                    threshold = float(np.percentile(scores_arr, 40)) if n > 1 else 0.3
                    merge_result = await merge_important_shots(
                        input_video=canonical_path,
                        shots_times=all_shots_times,
                        importance_scores=all_scores,
                        threshold=threshold,
                        max_duration=300,
                        return_manifest=True,
                    )
                    if isinstance(merge_result, tuple):
                        merged_video_path, merged_manifest = merge_result
                    else:
                        merged_video_path, merged_manifest = merge_result, []
                except Exception as e:
                    merged_video_path = None
                    merged_manifest = []
                    logger.warning(f"Failed to create merged video: {e}")

                chapters = _build_chapters_from_manifest(
                    merged_manifest=merged_manifest,
                    transcriptions=[_clean_transcript_text(t) for t in transcripts],
                    min_chapters=5,
                    max_chapters=12,
                )

                processing_completed_at = datetime.utcnow()
                processing_duration_sec = float(max(0.0, time.monotonic() - processing_started_mono))

                # Save summary record
                summary_kwargs = {
                    "summary_id": f"sum_{video_id}",
                    "video_id": video_id,
                    "type": "text_only",
                    "duration": 0,
                    "video_path": merged_video_path,
                    "summary_style": summary_type,
                    "text_summary_bullet": all_formats.get("bullet") or "",
                    "text_summary_structured": all_formats.get("structured") or "",
                    "text_summary_plain": all_formats.get("plain") or "",
                    "config_json": {
                        **config,
                        "processing_started_at": processing_started_at.isoformat() + "Z",
                        "processing_completed_at": processing_completed_at.isoformat() + "Z",
                        "processing_duration_sec": processing_duration_sec,
                        "long_video_mode": True,
                        "chunk_duration": CHUNK_DURATION,
                        "max_shots_per_chunk": LONG_MAX_SHOTS_PER_CHUNK,
                        "shot_threshold": LONG_SHOT_THRESHOLD,
                        "min_shot_duration": LONG_MIN_SHOT_DURATION,
                        "text_length": text_length,
                        "summary_type": summary_type,
                        "requested_format": summary_format,
                        "generated_formats": list(all_formats.keys()) or [summary_format],
                        "merged_video_enabled": merged_video_path is not None,
                        "merged_manifest": merged_manifest,
                        "evidence": evidence_items,
                        "chapters": chapters,
                    },
                }

                summary = Summary(**summary_kwargs)
                db.add(summary)
                video.status = "completed"
                await db.commit()

                await _send_log(
                    video_id,
                    f"✓ Processing complete (long video) in {format_duration(processing_duration_sec)}",
                    level="SUCCESS",
                    stage="completed",
                    progress=100,
                )

                # Cleanup even on the long-video early return path
                try:
                    uploaded_file_path = os.path.join(settings.UPLOAD_DIR, video.filename)
                    _cleanup_video_artifacts(
                        video_id=video_id,
                        uploaded_file_path=uploaded_file_path,
                        canonical_path=canonical_path,
                        keep_outputs=True,
                    )
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup error for {video_id} (long video): {cleanup_error}")
                return
            
            # 3. Shot Detection (use lower threshold 0.25 to detect more shots)
            video.status = "shot_detection"
            await db.commit()
            await _send_log(video_id, "Running shot detection", stage="shot_detection", progress=35)
            
            shots_times = await detect_shots(canonical_path, threshold=0.25)  # Lower threshold for more shots
            
            # Filter out very short shots which are usually UI noise/transitions.
            # Keep this slightly permissive; static slide videos often have few cuts already.
            min_shot_duration = 0.8
            original_count = len(shots_times)
            shots_times = [(s, e) for s, e in shots_times if (e - s) >= min_shot_duration]
            
            if len(shots_times) < original_count:
                logger.info(f"Filtered out {original_count - len(shots_times)} short shots (<{min_shot_duration}s)")
            
            # If shot detection yields too few segments (common for static slide/talking-head videos),
            # fall back to uniform time segmentation so chapters/evidence aren't starved.
            try:
                target_segments = int(math.ceil(float(video_duration_seconds) / 15.0)) if video_duration_seconds else 0
                target_segments = int(min(60, max(12, target_segments))) if target_segments > 0 else 12
                too_few = len(shots_times) < max(6, int(0.4 * target_segments))
                if too_few and float(video_duration_seconds) <= 30.0 * 60.0:
                    shots_times = _time_segment_shots(float(video_duration_seconds), target_segments)
                    await _send_log(
                        video_id,
                        f"Shot detection found too few cuts; using time segmentation ({len(shots_times)} segments)",
                        level="WARNING",
                        stage="shot_detection",
                        progress=45,
                    )
                else:
                    await _send_log(video_id, f"Detected {len(shots_times)} shots (filtered)", stage="shot_detection", progress=45)
            except Exception:
                await _send_log(video_id, f"Detected {len(shots_times)} shots (filtered)", stage="shot_detection", progress=45)
            
            # Create Shot objects
            shots_data = []
            for i, (start, end) in enumerate(shots_times):
                shot_id = f"{video_id}_{i:04d}"
                shots_data.append({
                    "shot_id": shot_id,
                    "video_id": video_id,
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": end - start
                })
                
            # Save shots to DB (simplified, usually bulk insert)
            # We'll skip DB insert for every shot here for brevity, 
            # but in prod we'd do it.
            
            # 4. Feature Extraction
            video.status = "feature_extraction"
            await db.commit()
            await _send_log(video_id, "Extracting frames and audio", stage="feature_extraction", progress=55)
            
            # Extract keyframes (multiple per shot) and track counts for aggregation
            keyframe_paths, frames_per_shot = await sample_frames_for_shots(
                canonical_path,
                shots_times,
                video_id,
                frames_per_shot=3,
            )

            # Persist keyframes so the frontend can display thumbnails even after cleanup
            keyframes_out_dir = os.path.join(settings.OUTPUT_DIR, "keyframes", video_id)
            os.makedirs(keyframes_out_dir, exist_ok=True)
            persisted_keyframes: list[str] = []
            offset = 0
            for i, count in enumerate(frames_per_shot):
                src = keyframe_paths[offset] if offset < len(keyframe_paths) else ""
                dst = os.path.join(keyframes_out_dir, f"shot_{i:04d}.jpg")
                try:
                    if src and os.path.exists(src):
                        shutil.copyfile(src, dst)
                        persisted_keyframes.append(dst)
                    else:
                        persisted_keyframes.append("")
                except Exception as e:
                    logger.warning(f"Failed to persist keyframe {i}: {e}")
                    persisted_keyframes.append("")
                offset += count
            
            # Extract audio for shots (optional, can be slow for many shots)
            # For prototype, let's extract full audio and process chunks in memory or 
            # just extract a few for demo. 
            # Let's extract per shot as per spec.
            audio_paths = []
            audio_dir = os.path.join(settings.PROCESSED_DIR, video_id, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            # Extract audio segments from video
            for i, (start, end) in enumerate(shots_times):
                path = os.path.join(audio_dir, f"shot_{i:04d}.wav")
                await extract_audio_segment(canonical_path, start, end, path)
                audio_paths.append(path)

            # Encoders
            vis_encoder = VisualEncoder()
            aud_encoder = AudioEncoder()

            # Visual: encode all frames then mean-pool per shot
            frame_feats = vis_encoder.encode(keyframe_paths)
            shot_embs = []
            offset = 0
            for n in frames_per_shot:
                if n <= 0:
                    shot_embs.append(torch.zeros(768))
                    continue
                slice_feats = frame_feats[offset : offset + n]
                offset += n
                if slice_feats.numel() == 0:
                    shot_embs.append(torch.zeros(768))
                else:
                    shot_embs.append(slice_feats.mean(dim=0))
            vis_feats = torch.stack(shot_embs) if shot_embs else torch.empty(0, 768)

            # Audio remains one segment per shot
            aud_feats = aud_encoder.encode(audio_paths)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Visual features shape: {vis_feats.shape}, Audio features shape: {aud_feats.shape}")
            
            # Extract handcrafted features for all shots
            handcrafted_feats = extract_handcrafted_features(
                shots_times=shots_times,
                video_duration=video_duration_seconds
            )  # (N, 14)
            
            # Generate text embeddings from transcriptions using existing service
            transcription_service = TranscriptionService(device="cpu")  # Force CPU
            text_embedder = TextEmbedder(device="cpu")  # Force CPU
            
            transcriptions = []
            for audio_path in audio_paths:
                try:
                    text = transcription_service.transcribe_audio(audio_path)
                    transcriptions.append(text if text else "")
                except Exception as e:
                    logger.warning(f"Failed to transcribe {audio_path}: {e}")
                    transcriptions.append("")
            
            text_feats_np = text_embedder.batch_encode(transcriptions)  # (N, 384)
            text_feats = torch.from_numpy(text_feats_np).float() if len(text_feats_np) > 0 else torch.zeros(len(transcriptions), 384)
            
            # Clear memory after text processing
            del transcription_service
            del text_embedder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Concatenate all features: 768 + 768 + 14 + 384 = 1934
            if vis_feats.shape[0] > 0 and aud_feats.shape[0] > 0:
                features = torch.cat([
                    vis_feats.to('cpu'),
                    aud_feats.to('cpu'),
                    handcrafted_feats,
                    text_feats
                ], dim=1)
            else:
                logger.warning("No features extracted successfully")
                features = torch.empty(len(keyframe_paths), 1934)
            
            logger.info(f"Feature composition: Visual=768, Audio=768, Handcrafted=14, Text=384, Total={features.shape[1]}")
            
            # 5. Graph & GNN with NEW InferenceService
            video.status = "gnn_inference"
            await db.commit()
            await _send_log(video_id, "Running GNN inference", stage="gnn_inference", progress=70)
            
            builder = GraphBuilder()
            graph_data = builder.build_graph(shots_data, features)
            
            # Get NEW inference service and process video end-to-end
            inference_service = get_inference_service()
            summary_type = config.get("summary_type", "balanced")
            text_length = config.get("text_length", "medium")
            summary_format = config.get("summary_format", "bullet")
            
            await _send_log(video_id, "Generating binary predictions and text summary", stage="gnn_inference", progress=75)
            
            # Run GNN scoring (text summaries generated later in one optimized pass)
            try:
                gnn_scores = inference_service.predict_importance_scores(
                    graph_data.x,
                    graph_data.edge_index
                )
                
                # Convert scores to list
                scores = gnn_scores.reshape(-1).tolist() if hasattr(gnn_scores, 'reshape') else list(gnn_scores)
                
                # Store importance scores in shot records
                for i, shot_data in enumerate(shots_data):
                    shot = Shot(**shot_data)
                    if i < len(scores):
                        shot.importance_score = scores[i]
                    if i < len(persisted_keyframes) and persisted_keyframes[i]:
                        shot.keyframe_path = persisted_keyframes[i]
                    db.add(shot)
                await db.commit()
                
                await _send_log(video_id, "✓ GNN inference successful", stage="gnn_inference", progress=80)
                used_fallback = False
                
            except Exception as gnn_error:
                logger.error(f"GNN processing failed for {video_id}: {gnn_error}", exc_info=True)
                await _send_log(video_id, f"GNN inference failed: {gnn_error}. Attempting Gemini fallback...", level="WARNING", stage="gnn_inference", progress=75)
                
                # Fallback to Gemini API for summarization
                gemini = get_gemini_summarizer()
                if gemini.is_available():
                    await _send_log(video_id, "⚠️  Switching to Gemini API fallback for text summarization", level="WARNING", stage="gnn_fallback", progress=76)
                    # Request a plain summary once, then format locally into bullet/structured.
                    gemini_text, gemini_metadata = gemini.summarize_video_from_path(
                        video_path=canonical_path,
                        summary_type=summary_type,
                        text_length=text_length,
                        summary_format="plain"
                    )
                    
                    if gemini_text:
                        await _send_log(video_id, "✓ Gemini API fallback successful - using alternative summarization", level="WARNING", stage="gnn_fallback", progress=82)
                        print(f"[FALLBACK USED] Video {video_id}: GNN failed, Gemini API used as fallback")
                        logger.info(f"[FALLBACK USED] Video {video_id}: GNN failed, using Gemini API. Metadata: {gemini_metadata}")
                        used_fallback = True
                        
                        # Create synthetic scores since GNN failed
                        scores = [0.5] * len(shots_data)  # Default neutral score
                        
                        # Store shot records with synthetic scores
                        for i, shot_data in enumerate(shots_data):
                            shot = Shot(**shot_data)
                            shot.importance_score = 0.5  # Neutral score
                            db.add(shot)
                        await db.commit()
                    else:
                        error_msg = gemini_metadata.get("error", "Unknown Gemini error")
                        await _send_log(video_id, f"Gemini fallback also failed: {error_msg}", level="ERROR", stage="gnn_fallback", progress=80)
                        raise Exception(f"Both GNN and Gemini failed: {gnn_error} | {error_msg}")
                else:
                    await _send_log(video_id, "Gemini API not available. Cannot proceed without GNN or Gemini fallback.", level="ERROR", stage="gnn_fallback", progress=80)
                    raise Exception(f"GNN failed and Gemini API not available: {gnn_error}")
            
            # 6. Text Summary Generation - MAIN FEATURE
            await _send_log(video_id, "Generating text summaries in multiple formats", stage="summarization", progress=85)

            all_formats: dict[str, str] = {}
            evidence_items: list[dict] = []
            used_gemini_text = False
            if used_fallback:
                # Gemini already produced a plain summary; format locally for the other views
                summarizer = inference_service.manager.get_summarizer()
                if summary_format == "plain":
                    all_formats["plain"] = gemini_text
                else:
                    formatted = summarizer.format_text(
                        gemini_text,
                        summary_format,
                        summary_type=summary_type,
                        text_length=text_length,
                    )
                    all_formats[summary_format] = formatted
            else:
                # Transcribe once so we can attach evidence snippets to bullets
                transcriber = inference_service.manager.get_whisper()
                transcripts: list[str] = []
                for audio_path in audio_paths:
                    audio_path_obj = Path(audio_path) if isinstance(audio_path, str) else audio_path
                    try:
                        if audio_path_obj.exists():
                            t = transcriber.transcribe(audio_path_obj, cache_dir=None)
                            transcripts.append(_clean_transcript_text((t or "").strip()))
                        else:
                            transcripts.append("")
                    except Exception as e:
                        logger.warning(f"Transcription failed for {audio_path_obj}: {e}")
                        transcripts.append("")

                summarizer = inference_service.manager.get_summarizer()
                try:
                    nonempty = sum(1 for t in transcripts if t and t.strip())
                    total_words = sum(_safe_word_count(t) for t in transcripts if t and t.strip())
                    logger.info(f"Transcript stats: nonempty={nonempty}/{len(transcripts)}, total_words={total_words}")
                    if total_words < 25:
                        logger.warning(
                            "Transcript signal is sparse/noisy; summary may be shorter or rely more on visual cues"
                        )
                except Exception:
                    pass

                # Optional Gemini override for text summary (does not affect GNN scoring / merged video)
                try:
                    gemini_enabled = bool(getattr(settings, "ENABLE_GEMINI_SUMMARIZER", False))
                    gemini_force = bool(getattr(settings, "FORCE_GEMINI_SUMMARIZER", False))
                except Exception:
                    gemini_enabled = False
                    gemini_force = False

                if gemini_enabled and (gemini_force or ("total_words" in locals() and int(total_words) < 60)):
                    gemini = get_gemini_summarizer()
                    if gemini.is_available():
                        await _send_log(video_id, "Using Gemini for text summary (override enabled)", level="WARNING", stage="summarization", progress=86)
                        gem_text, gem_meta = gemini.summarize_video_from_path(
                            video_path=canonical_path,
                            summary_type=summary_type,
                            text_length=text_length,
                            summary_format="plain",
                        )
                        if gem_text:
                            if summary_format == "plain":
                                all_formats["plain"] = gem_text
                            else:
                                formatted = summarizer.format_text(
                                    gem_text,
                                    summary_format,
                                    summary_type=summary_type,
                                    text_length=text_length,
                                )
                                all_formats[summary_format] = formatted
                            used_gemini_text = True
                            await _send_log(video_id, "✓ Gemini text summary generated", level="WARNING", stage="summarization", progress=87)
                        else:
                            logger.warning(f"Gemini override enabled but failed: {gem_meta}")

                if used_gemini_text:
                    # Still build evidence items from transcripts + scores below.
                    pass
                k_ratio = max(0.0, min(1.0, float(getattr(settings, "TOPK_RATIO", 0.15))))
                n = len(transcripts)
                # Ensure we pick a reasonable minimum number of shots for short videos,
                # otherwise evidence/chapters often collapse to 1-3 items.
                min_top_k = 5
                try:
                    min_top_k = int(math.ceil(float(video_duration_seconds) / 60.0))  # ~1 per minute
                    min_top_k = int(min(25, max(5, min_top_k)))
                except Exception:
                    min_top_k = 5
                min_top_k = int(min(max(1, n), min_top_k))
                top_k = max(min_top_k, int(__import__('numpy').ceil(k_ratio * n))) if n > 0 else 1
                top_k = int(min(top_k, n)) if n > 0 else 1

                base_scores_list = gnn_scores.tolist() if hasattr(gnn_scores, 'tolist') else list(gnn_scores)
                scores_for_summary = _adjust_scores_for_summary_type(
                    summary_type=summary_type,
                    base_scores=[float(s) for s in base_scores_list],
                    transcripts=transcripts,
                    shots_times=shots_times,
                    audio_paths=[str(p) for p in audio_paths],
                    keyframe_paths=[str(p) for p in persisted_keyframes],
                )
                if not used_gemini_text:
                    all_formats = summarizer.summarize_all_formats(
                        transcripts=transcripts,
                        gnn_scores=scores_for_summary,
                        summary_type=summary_type,
                        text_length=text_length,
                        top_k=top_k,
                        video_path=canonical_path,
                        formats=[summary_format],
                    )

                # Build evidence items by aligning bullet lines to top-K shot indices.
                scores_list = scores_for_summary
                scores_arr = __import__('numpy').array(scores_list, dtype=float)
                if scores_arr.size > 0:
                    top_indices = scores_arr.argsort()[::-1][: min(top_k, scores_arr.size)]
                    top_indices_sorted = sorted([int(i) for i in top_indices])
                else:
                    top_indices_sorted = []

                for j, shot_idx in enumerate(top_indices_sorted):
                    start_sec, end_sec = shots_times[shot_idx] if shot_idx < len(shots_times) else (0.0, 0.0)
                    shot_id = f"{video_id}_{shot_idx:04d}"
                    transcript_snippet = (transcripts[shot_idx] if shot_idx < len(transcripts) else "").strip()
                    if len(transcript_snippet) > 260:
                        transcript_snippet = transcript_snippet[:260] + "…"
                    thumb_path = persisted_keyframes[shot_idx] if shot_idx < len(persisted_keyframes) else ""

                    # Explainability signals (cheap but informative)
                    duration = float(end_sec - start_sec) if end_sec and start_sec else 0.0
                    wc = _safe_word_count(transcripts[shot_idx] if shot_idx < len(transcripts) else "")
                    transcript_density = (wc / max(0.25, duration)) if duration > 0 else float(wc)
                    audio_rms = _audio_rms_energy(audio_paths[shot_idx] if shot_idx < len(audio_paths) else None)
                    prev_kf = persisted_keyframes[shot_idx - 1] if shot_idx - 1 >= 0 and shot_idx - 1 < len(persisted_keyframes) else None
                    cur_kf = persisted_keyframes[shot_idx] if shot_idx < len(persisted_keyframes) else None
                    scene_change = _histogram_delta(prev_kf, cur_kf) if shot_idx > 0 else None

                    # Motion is computed only for selected shots (ffmpeg 2-frame diff)
                    motion_tmp = os.path.join(settings.TEMP_DIR, video_id, "motion", f"shot_{shot_idx:04d}")
                    motion = await _motion_estimate(canonical_path, float(start_sec), float(end_sec), motion_tmp)

                    neighbors = _graph_neighbors(graph_data, int(shot_idx), max_neighbors=8)

                    signals = {
                        "motion": motion,
                        "audio_rms": audio_rms,
                        "scene_change": scene_change,
                        "transcript_density": float(transcript_density),
                        "duration_sec": float(duration),
                    }

                    heading, description = _evidence_heading_description(
                        transcript_snippet=transcript_snippet,
                        shot_index=int(shot_idx),
                        signals=signals,
                    )
                    justification = _evidence_justification(
                        score=float(scores_list[shot_idx]) if shot_idx < len(scores_list) and scores_list[shot_idx] is not None else None,
                        signals=signals,
                        neighbors=neighbors,
                        normalized_signals=None,
                    )

                    evidence_items.append({
                        "index": j,
                        "shot_index": int(shot_idx),
                        "shot_id": shot_id,
                        "orig_start": float(start_sec),
                        "orig_end": float(end_sec),
                        "score": float(scores_list[shot_idx]) if shot_idx < len(scores_list) else None,
                        "bullet": heading,
                        "description": description,
                        "justification": justification,
                        "transcript_snippet": transcript_snippet,
                        "thumbnail_path": thumb_path,
                        "signals": signals,
                        "neighbors": neighbors,
                    })

            text_summary = all_formats.get(summary_format) or all_formats.get("bullet") or ""
            logger.info(
                f"✓ Generated summaries: bullet={len(all_formats.get('bullet',''))} chars, "
                f"structured={len(all_formats.get('structured',''))} chars, plain={len(all_formats.get('plain',''))} chars"
            )
            
            await _send_log(video_id, "Summary generation complete", stage="summarization", progress=90)
            
            # 7. Generate merged video of important shots
            await _send_log(video_id, "Creating merged video of important shots", stage="video_merge", progress=92)
            try:
                # Use adaptive threshold based on score distribution
                scores_array = gnn_scores if not used_fallback else scores
                if isinstance(scores_array, torch.Tensor):
                    scores_array = scores_array.cpu().numpy()
                else:
                    scores_array = np.array(scores_array)

                # Calculate threshold as 30th percentile to select top 70% of shots
                # This gives more shots for a richer compilation (10min video = ~50-70 shots)
                if len(scores_array) > 0:
                    threshold = float(np.percentile(scores_array, 30)) if len(scores_array) > 1 else 0.3
                else:
                    threshold = 0.3
                
                logger.info(f"Adaptive threshold (30th percentile): {threshold:.3f}, will select ~{int(len(scores_array) * 0.7)} shots")
                
                merge_result = await merge_important_shots(
                    input_video=canonical_path,
                    shots_times=shots_times,
                    importance_scores=scores_array.tolist() if hasattr(scores_array, 'tolist') else scores_array,
                    threshold=threshold,
                    max_duration=300,  # Max 5 minutes for merged video
                    return_manifest=True
                )

                if isinstance(merge_result, tuple):
                    merged_video_path, merged_manifest = merge_result
                else:
                    merged_video_path, merged_manifest = merge_result, []

                # Persist merged video into outputs so we can safely clear temp without breaking downloads/history
                if merged_video_path and os.path.exists(merged_video_path):
                    merged_out_dir = os.path.join(settings.OUTPUT_DIR, "merged")
                    os.makedirs(merged_out_dir, exist_ok=True)
                    merged_out_path = os.path.join(merged_out_dir, f"{video_id}.mp4")
                    try:
                        try:
                            shutil.move(merged_video_path, merged_out_path)
                        except Exception:
                            # Cross-device move fallback
                            shutil.copyfile(merged_video_path, merged_out_path)
                            _safe_remove_path(merged_video_path)
                        merged_video_path = merged_out_path
                    except Exception as persist_err:
                        logger.warning(f"Failed to persist merged video to outputs: {persist_err}")

                chapters = _build_chapters_from_manifest(
                    merged_manifest=merged_manifest,
                    transcriptions=[_clean_transcript_text(t) for t in transcriptions],
                    min_chapters=5,
                    max_chapters=12,
                )
                
                await _send_log(video_id, f"✓ Merged video created: {os.path.basename(merged_video_path)}", stage="video_merge", progress=93)
                logger.info(f"Merged video: {merged_video_path}")
            except Exception as e:
                merged_video_path = None
                merged_manifest = []
                logger.warning(f"Failed to create merged video: {e}")
                await _send_log(video_id, f"⚠️  Merged video skipped: {str(e)[:100]}", level="WARNING", stage="video_merge", progress=93)
            
            # 8. Finalize
            video.status = "completed"
            await _send_log(video_id, "Processing complete", stage="completed", progress=95)
            
            # Save summary record with all formats
            fallback_note = " [GEMINI FALLBACK]" if used_fallback else ""
            text_summary_preview = text_summary[:150] if text_summary else "No summary"
            
            # Log text summary preview
            await _send_log(
                video_id,
                f"✓ Text summary generated{fallback_note}: {text_summary_preview}...",
                level="SUCCESS",
                stage="completed",
                progress=98
            )
            
            # Create and save summary record
            processing_completed_at = datetime.utcnow()
            processing_duration_sec = float(max(0.0, time.monotonic() - processing_started_mono))

            summary_kwargs = {
                "summary_id": f"sum_{video_id}",
                "video_id": video_id,
                "type": "text_only",
                "duration": 0,
                "video_path": merged_video_path,  # Store merged video path
                "summary_style": summary_type,
                "text_summary_bullet": all_formats.get("bullet") or "",
                "text_summary_structured": all_formats.get("structured") or "",
                "text_summary_plain": all_formats.get("plain") or "",
                "config_json": {
                    **config,
                    "processing_started_at": processing_started_at.isoformat() + "Z",
                    "processing_completed_at": processing_completed_at.isoformat() + "Z",
                    "processing_duration_sec": processing_duration_sec,
                    "fallback_used": used_fallback,
                    "text_length": text_length,
                    "summary_type": summary_type,
                    "requested_format": summary_format,
                        "generated_formats": list(all_formats.keys()) or [summary_format],
                    "merged_video_enabled": merged_video_path is not None,
                    "merged_manifest": merged_manifest,
                    "evidence": evidence_items,
                    "chapters": chapters if 'chapters' in locals() else [],
                }
            }
            
            summary = Summary(**summary_kwargs)
            db.add(summary)
            await db.commit()
            
            await _send_log(
                video_id,
                f"✓ Processing complete in {format_duration(processing_duration_sec)}",
                level="SUCCESS",
                stage="completed",
                progress=100
            )
            
            # 9. Cleanup temporary files
            await _send_log(video_id, "Cleaning up temporary files...", stage="cleanup", progress=99)
            try:
                uploaded_file_path = os.path.join(settings.UPLOAD_DIR, video.filename)
                _cleanup_video_artifacts(
                    video_id=video_id,
                    uploaded_file_path=uploaded_file_path,
                    canonical_path=canonical_path,
                    keep_outputs=True,
                )
                
                await _send_log(video_id, "✓ Temporary files cleaned up", stage="cleanup", progress=100)
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error for {video_id}: {cleanup_error}")
                await _send_log(video_id, f"⚠️  Partial cleanup: {str(cleanup_error)[:100]}", level="WARNING", stage="cleanup", progress=100)

            
        except Exception as e:
            logger.error(f"Task failed for video {video_id}: {e}", exc_info=True)
            print(f"Task failed: {e}")
            video.status = "failed"
            await db.commit()
            try:
                processing_duration_sec = float(max(0.0, time.monotonic() - processing_started_mono))
                await _send_log(
                    video_id,
                    f"Processing failed after {format_duration(processing_duration_sec)}: {e}",
                    level="ERROR",
                    stage="failed",
                    progress=100,
                )
            except Exception:
                await _send_log(video_id, f"Processing failed: {e}", level="ERROR", stage="failed", progress=100)
