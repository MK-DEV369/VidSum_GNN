from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
import shutil
import os
import uuid
import json
from datetime import datetime
import time
import hashlib
import subprocess

from vidsum_gnn.db.client import get_db
from vidsum_gnn.db.models import Video, Summary, Shot
from vidsum_gnn.core.config import settings
from vidsum_gnn.api.tasks import process_video_task
from vidsum_gnn.utils.logging import get_logger
from pydantic import BaseModel

router = APIRouter()
logger = get_logger(__name__)

# ============ Pydantic Models for Request/Response ============

class ProcessRequest(BaseModel):
    """Request body for manual video processing"""
    target_duration: int = 60
    selection_method: str = "greedy"
    text_length: str = "medium"
    summary_format: str = "bullet"
    summary_type: str = "balanced"
    generate_video: bool = False

class UploadOptions(BaseModel):
    """Video upload configuration options"""
    text_length: str = "medium"
    summary_format: str = "bullet"
    summary_type: str = "balanced"
    generate_video: bool = False

class SummaryConfigResponse(BaseModel):
    """Available summary configuration options"""
    text_lengths: Dict[str, str]
    summary_formats: Dict[str, str]
    summary_types: Dict[str, str]
    default_options: Dict[str, str]

class TextSummaryResponse(BaseModel):
    """Response for text summary retrieval"""
    video_id: str
    summary: str
    format: str
    style: str
    generated_at: Optional[str] = None
    processing_started_at: Optional[str] = None
    processing_completed_at: Optional[str] = None
    processing_duration_sec: Optional[float] = None

class SummaryResultResponse(BaseModel):
    """Complete summary result"""
    video_id: str
    status: str
    text_summaries: Dict[str, str]
    summary_type: str
    fallback_used: bool
    generated_at: Optional[str] = None
    processing_started_at: Optional[str] = None
    processing_completed_at: Optional[str] = None
    processing_duration_sec: Optional[float] = None


class TtsVoice(BaseModel):
    short_name: str
    friendly_name: Optional[str] = None
    locale: Optional[str] = None
    gender: Optional[str] = None


class TtsVoicesResponse(BaseModel):
    voices: List[TtsVoice]


class TtsRequest(BaseModel):
    text: str
    voice: str
    rate: float = 1.0
    video_id: Optional[str] = None


class TtsResponse(BaseModel):
    audio_url: str
    voice: str
    rate: float


_TTS_VOICES_CACHE: dict = {"ts": 0.0, "voices": []}


def _gtts_voice_presets() -> list[TtsVoice]:
    """A small set of free Google TTS (gTTS) presets.

    gTTS supports many languages and a few accent variants via `tld`.
    We expose them as pseudo-voices so the frontend can offer languages/accents
    even if edge-tts is unavailable.
    """
    presets: list[tuple[str, str, str]] = [
        # (language, tld, label)
        ("en", "com", "English (US)") ,
        ("en", "co.uk", "English (UK)"),
        ("en", "com.au", "English (AU)"),
        ("en", "ca", "English (CA)"),
        ("en", "co.in", "English (IN)"),
        ("hi", "co.in", "Hindi (IN)"),
        ("ta", "co.in", "Tamil (IN)"),
        ("te", "co.in", "Telugu (IN)"),
        ("ml", "co.in", "Malayalam (IN)"),
        ("kn", "co.in", "Kannada (IN)"),
        ("bn", "co.in", "Bengali (IN)"),
        ("ur", "com", "Urdu"),
        ("es", "com", "Spanish"),
        ("fr", "fr", "French"),
        ("de", "de", "German"),
        ("it", "it", "Italian"),
        ("pt", "com.br", "Portuguese (BR)"),
        ("ja", "co.jp", "Japanese"),
        ("ko", "co.kr", "Korean"),
        ("zh-CN", "com", "Chinese (Simplified)"),
        ("ar", "com", "Arabic"),
    ]
    out: list[TtsVoice] = []
    for lang, tld, label in presets:
        out.append(
            TtsVoice(
                short_name=f"gtts:{lang}:{tld}",
                friendly_name=f"Google TTS – {label}",
                locale=lang,
                gender=None,
            )
        )
    return out


def _rate_to_edge(rate: float) -> str:
    """Convert a numeric speaking rate (e.g. 0.9, 1.0, 1.25) to edge-tts format."""
    try:
        r = float(rate)
    except Exception:
        r = 1.0
    r = max(0.5, min(2.0, r))
    pct = int(round((r - 1.0) * 100.0))
    pct = max(-50, min(100, pct))
    return f"{pct:+d}%"

async def broadcast_log(request: Request, video_id: str, log_data: dict):
    """Broadcast log to connected WebSocket clients"""
    try:
        from vidsum_gnn.api.main import manager
        await manager.send_log(video_id, log_data)
    except Exception as e:
        logger.error(f"Failed to broadcast log: {e}")

# ============ Endpoints ============

@router.get("/config")
async def get_summary_config() -> SummaryConfigResponse:
    """
    Get available summarization configuration options.
    Useful for frontend to populate dropdown menus.
    
    Returns:
        Configuration with all available options and defaults
    """
    return SummaryConfigResponse(
        text_lengths={
            "short": "Short (50-100 words) - Brief overview",
            "medium": "Medium (100-200 words) - Standard summary",
            "long": "Long (200-400 words) - Detailed summary"
        },
        summary_formats={
            "bullet": "Bullet Points (• format) - Quick scanning",
            "structured": "Structured (Sections) - Organized layout",
            "plain": "Plain Text (Paragraphs) - Natural reading"
        },
        summary_types={
            "balanced": "Balanced - Mix of visual and audio elements",
            "visual_priority": "Visual Priority - Focus on what you see",
            "audio_priority": "Audio Priority - Focus on dialogue/narration",
            "highlights": "Highlights - Most important moments only"
        },
        default_options={
            "text_length": "medium",
            "summary_format": "bullet",
            "summary_type": "balanced"
        }
    )

@router.post("/upload")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    text_length: str = Form("medium"),
    summary_format: str = Form("bullet"),
    summary_type: str = Form("balanced"),
    generate_video: bool = Form(False),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and process a video file.
    
    Parameters:
    - file: Video file to process
    - text_length: "short" | "medium" | "long"
    - summary_format: "bullet" | "structured" | "plain"
    - summary_type: "balanced" | "visual_priority" | "audio_priority" | "highlights"
    - generate_video: Whether to generate a video summary (default: false)
    
    Returns:
    - video_id: Unique identifier for tracking progress
    - status: "queued" (processing will begin shortly)
    """
    video_id = str(uuid.uuid4())
    filename = f"{video_id}_{file.filename}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    
    # Validate parameters
    valid_lengths = ["short", "medium", "long"]
    valid_formats = ["bullet", "structured", "plain"]
    valid_types = ["balanced", "visual_priority", "audio_priority", "highlights"]
    
    if text_length not in valid_lengths:
        raise HTTPException(status_code=400, detail=f"Invalid text_length: {text_length}")
    if summary_format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"Invalid summary_format: {summary_format}")
    if summary_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid summary_type: {summary_type}")
    
    # Log upload start
    await broadcast_log(request, video_id, {
        "timestamp": datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Starting upload: {file.filename} ({file.size or 0} bytes)",
        "stage": "UPLOAD",
        "progress": 5
    })
    
    try:
        # Create upload directory if needed
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Save file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved: {filepath} ({os.path.getsize(filepath)} bytes)")
        
        # Create DB entry
        video = Video(
            video_id=video_id,
            filename=filename,
            status="queued",
            target_duration=60,
            selection_method="greedy"
        )
        db.add(video)
        await db.commit()
        
        # Log upload complete
        await broadcast_log(request, video_id, {
            "timestamp": datetime.now().isoformat(),
            "level": "SUCCESS",
            "message": f"Upload complete. Starting processing...",
            "stage": "UPLOAD",
            "progress": 15,
            "config": {
                "text_length": text_length,
                "summary_format": summary_format,
                "summary_type": summary_type,
                "generate_video": generate_video
            }
        })
        
        # Start background processing
        background_tasks.add_task(
            process_video_task,
            video_id,
            {
                "text_length": text_length,
                "summary_format": summary_format,
                "summary_type": summary_type,
                "generate_video": generate_video,
                "target_duration": 60,
                "selection_method": "greedy"
            }
        )
        
        return {
            "video_id": video_id,
            "status": "queued",
            "message": "Video uploaded successfully. Processing started.",
            "config": {
                "text_length": text_length,
                "summary_format": summary_format,
                "summary_type": summary_type
            }
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Upload error: {error_msg}", exc_info=True)
        
        await broadcast_log(request, video_id, {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "message": f"Upload failed: {error_msg}",
            "stage": "UPLOAD"
        })
        
        raise HTTPException(status_code=400, detail=f"Upload failed: {error_msg}")

@router.post("/process/{video_id}")
async def process_video(
    video_id: str, 
    request: ProcessRequest, 
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Manually trigger processing for an uploaded video"""
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    logger.info(f"Manual processing triggered for {video_id}")
    background_tasks.add_task(process_video_task, video_id, request.dict())
    
    return {
        "message": "Processing started",
        "task_id": video_id,
        "config": request.dict()
    }

@router.get("/status/{video_id}")
async def get_status(video_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get processing status of a video.
    
    Returns:
    - status: "queued" | "preprocessing" | "shot_detection" | "feature_extraction" | 
              "gnn_inference" | "assembling" | "completed" | "failed"
    """
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return {
        "video_id": video_id,
        "status": video.status,
        "target_duration": video.target_duration,
        "selection_method": video.selection_method,
        "created_at": video.created_at
    }

@router.get("/results/{video_id}")
async def get_results(video_id: str, db: AsyncSession = Depends(get_db)) -> SummaryResultResponse:
    """
    Get complete summary results for a video.
    
    Returns all three formats of the summary plus metadata.
    """
    result = await db.execute(select(Summary).where(Summary.video_id == video_id))
    summary = result.scalar_one_or_none()
    
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    # Parse config to get original summary type and fallback status
    config = summary.config_json or {}
    fallback_used = config.get("fallback_used", False)
    summary_type = config.get("summary_type", "balanced")
    
    return SummaryResultResponse(
        video_id=video_id,
        status="completed",
        text_summaries={
            "bullet": summary.text_summary_bullet or "",
            "structured": summary.text_summary_structured or "",
            "plain": summary.text_summary_plain or ""
        },
        summary_type=summary_type,
        fallback_used=fallback_used,
        generated_at=summary.generated_at.isoformat() if summary.generated_at else None,
        processing_started_at=config.get("processing_started_at"),
        processing_completed_at=config.get("processing_completed_at"),
        processing_duration_sec=config.get("processing_duration_sec"),
    )

@router.get("/summary/{video_id}/text")
async def get_text_summary(
    video_id: str,
    format: str = "bullet",
    db: AsyncSession = Depends(get_db)
) -> TextSummaryResponse:
    """
    Get text summary in a specific format.
    
    Parameters:
    - format: "bullet" | "structured" | "plain"
    
    Returns:
    - summary: The formatted text summary
    - format: The requested format
    - style: The summary type used (balanced, visual, etc.)
    """
    if format not in ["bullet", "structured", "plain"]:
        raise HTTPException(status_code=400, detail=f"Invalid format: {format}")
    
    result = await db.execute(select(Summary).where(Summary.video_id == video_id))
    summary = result.scalar_one_or_none()
    
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    
    # Map format to database column
    summary_map = {
        "bullet": summary.text_summary_bullet,
        "structured": summary.text_summary_structured,
        "plain": summary.text_summary_plain
    }
    
    # Get requested format, fallback to any available format if not found
    requested_summary = summary_map.get(format)
    if not requested_summary:
        # Try to find any non-empty summary
        for fmt, text in summary_map.items():
            if text:
                requested_summary = text
                format = fmt  # Update format to reflect what we're actually returning
                break
    
    if not requested_summary:
        raise HTTPException(status_code=404, detail="No text summary available")

    config = summary.config_json or {}
    
    return TextSummaryResponse(
        video_id=video_id,
        summary=requested_summary,
        format=format,
        style=summary.summary_style,
        generated_at=summary.generated_at.isoformat() if summary.generated_at else None,
        processing_started_at=config.get("processing_started_at"),
        processing_completed_at=config.get("processing_completed_at"),
        processing_duration_sec=config.get("processing_duration_sec"),
    )


@router.get("/summary/{video_id}/evidence")
async def get_summary_evidence(
    video_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Return evidence-linked items for the latest summary.

    This enables the frontend to render clickable bullets that seek into the merged
    summary video timeline (using merged_manifest offsets).
    """
    result = await db.execute(select(Summary).where(Summary.video_id == video_id))
    summary = result.scalar_one_or_none()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")

    config = summary.config_json or {}
    evidence = config.get("evidence", []) or []
    manifest = config.get("merged_manifest", []) or []

    manifest_by_shot = {
        int(m.get("shot_index")): m
        for m in manifest
        if isinstance(m, dict) and m.get("shot_index") is not None
    }

    items: List[Dict[str, Any]] = []
    for ev in evidence:
        shot_index = ev.get("shot_index")
        m = manifest_by_shot.get(int(shot_index)) if shot_index is not None else None

        items.append({
            "index": ev.get("index"),
            "bullet": ev.get("bullet"),
            "shot_index": shot_index,
            "shot_id": ev.get("shot_id"),
            "orig_start": ev.get("orig_start"),
            "orig_end": ev.get("orig_end"),
            "merged_start": m.get("merged_start") if isinstance(m, dict) else None,
            "merged_end": m.get("merged_end") if isinstance(m, dict) else None,
            "score": ev.get("score"),
            "transcript_snippet": ev.get("transcript_snippet"),
            "thumbnail_url": f"/api/keyframe/{video_id}/{int(shot_index)}" if shot_index is not None else None,
            "signals": ev.get("signals"),
            "neighbors": ev.get("neighbors"),
        })

    return {
        "video_id": video_id,
        "summary_style": summary.summary_style,
        "text_length": config.get("text_length"),
        "requested_format": config.get("requested_format"),
        "merged_video_enabled": config.get("merged_video_enabled", False),
        "items": items,
    }


@router.get("/summary/{video_id}/chapters")
async def get_summary_chapters(
    video_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Return YouTube-style chapters for the latest summary (merged timeline)."""
    result = await db.execute(select(Summary).where(Summary.video_id == video_id))
    summary = result.scalar_one_or_none()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")

    config = summary.config_json or {}
    chapters = config.get("chapters", []) or []
    return {
        "video_id": video_id,
        "merged_video_enabled": config.get("merged_video_enabled", False),
        "items": chapters,
    }


@router.get("/keyframe/{video_id}/{shot_index}")
async def get_keyframe(
    video_id: str,
    shot_index: int,
    db: AsyncSession = Depends(get_db)
):
    """Serve a persisted keyframe image for a given shot index."""
    shot_id = f"{video_id}_{int(shot_index):04d}"
    shot = await db.get(Shot, shot_id)
    if not shot or not shot.keyframe_path:
        raise HTTPException(status_code=404, detail="Keyframe not found")

    if not os.path.exists(shot.keyframe_path):
        raise HTTPException(status_code=404, detail="Keyframe file missing")

    return FileResponse(shot.keyframe_path, media_type="image/jpeg")

@router.get("/shot-scores/{video_id}")
async def get_shot_scores(video_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get GNN importance scores for all shots.
    
    Useful for visualizing which parts of the video were considered important.
    """
    result = await db.execute(select(Shot).where(Shot.video_id == video_id))
    shots = result.scalars().all()
    
    return {
        "video_id": video_id,
        "total_shots": len(shots),
        "shots": [
            {
                "shot_id": shot.shot_id,
                "start_sec": shot.start_sec,
                "end_sec": shot.end_sec,
                "duration_sec": shot.duration_sec,
                "importance_score": shot.importance_score
            }
            for shot in shots
        ]
    }

@router.get("/videos")
async def list_videos(
    status: Optional[str] = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """
    List all uploaded videos with pagination.
    
    Parameters:
    - status: Filter by status (optional)
    - limit: Maximum number of videos to return
    """
    query = select(Video)
    
    if status:
        query = query.where(Video.status == status)
    
    query = query.limit(limit)
    result = await db.execute(query)
    videos = result.scalars().all()
    
    return {
        "count": len(videos),
        "videos": [
            {
                "video_id": v.video_id,
                "filename": v.filename,
                "status": v.status,
                "created_at": v.created_at.isoformat() if v.created_at else None
            }
            for v in videos
        ]
    }

@router.get("/download/{video_id}")
async def download_merged_video(video_id: str, db: AsyncSession = Depends(get_db)):
    """Download the merged video compilation of important shots"""
    result = await db.execute(
        select(Summary).where(Summary.video_id == video_id)
    )
    summary = result.scalar_one_or_none()
    
    if not summary or not summary.video_path:
        logger.error(f"No summary or video_path for {video_id}")
        raise HTTPException(status_code=404, detail="Summary video not found")
    
    from pathlib import Path
    video_path = Path(summary.video_path)
    
    logger.info(f"[Download] Video ID: {video_id}, Path: {summary.video_path}, Exists: {video_path.exists()}")
    
    if not video_path.exists():
        logger.error(f"Video file does not exist: {video_path}")
        raise HTTPException(status_code=404, detail="Summary video file not found")
    
    logger.info(f"[Download] Serving video: {video_path}")
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"summary_{video_id}.mp4"
    )

@router.delete("/{video_id}")
async def delete_video(video_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a video and its associated data"""
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        # Delete associated files
        filepath = os.path.join(settings.UPLOAD_DIR, video.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Delete from database
        await db.delete(video)
        
        # Also delete associated summaries and shots (cascades)
        await db.execute(select(Summary).where(Summary.video_id == video_id))
        await db.execute(select(Shot).where(Shot.video_id == video_id))
        
        await db.commit()
        
        logger.info(f"Deleted video: {video_id}")
        
        return {
            "message": "Video deleted successfully",
            "video_id": video_id
        }
    except Exception as e:
        logger.error(f"Delete error for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@router.get("/tts/voices")
async def get_tts_voices() -> TtsVoicesResponse:
    """Return a list of available AI voices for backend TTS.

    Uses edge-tts (Microsoft neural voices). This avoids browser speechSynthesis.
    """
    # Cache for a day to avoid repeated remote lookups
    now = time.time()
    if _TTS_VOICES_CACHE.get("voices") and (now - float(_TTS_VOICES_CACHE.get("ts") or 0.0)) < 24 * 3600:
        return TtsVoicesResponse(voices=_TTS_VOICES_CACHE["voices"])

    out: list[TtsVoice] = []

    # Always include a few free Google TTS presets so the UI has voices even if edge-tts fails.
    out.extend(_gtts_voice_presets())

    # Try edge-tts (Microsoft neural voices) for richer, multi-voice selection.
    try:
        import edge_tts

        voices_raw = await edge_tts.list_voices()
        edge_out: list[TtsVoice] = []
        for v in voices_raw or []:
            try:
                locale = v.get("Locale") or v.get("locale")
                short = v.get("ShortName") or v.get("shortName")
                if not short:
                    continue
                edge_out.append(
                    TtsVoice(
                        short_name=str(short),
                        friendly_name=v.get("FriendlyName") or v.get("friendlyName"),
                        locale=locale,
                        gender=v.get("Gender") or v.get("gender"),
                    )
                )
            except Exception:
                continue

        # Keep list reasonable (but multi-language) to avoid a huge dropdown.
        edge_out = sorted(edge_out, key=lambda x: (x.locale or "", x.gender or "", x.short_name))
        edge_out = edge_out[:120]
        out.extend(edge_out)
    except Exception:
        # edge-tts not installed or voice listing failed; Google presets still work.
        pass

    # Final sort: group by provider then locale
    out = sorted(
        out,
        key=lambda x: (
            0 if str(x.short_name).startswith("gtts:") else 1,
            x.locale or "",
            x.gender or "",
            x.short_name,
        ),
    )

    _TTS_VOICES_CACHE["ts"] = now
    _TTS_VOICES_CACHE["voices"] = out
    return TtsVoicesResponse(voices=out)


@router.post("/tts", response_model=TtsResponse)
async def generate_tts(req: TtsRequest):
    """Generate an MP3 for the provided text and return a URL to play it."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if len(text) > 12000:
        # Keep generation bounded
        text = text[:12000]

    voice = (req.voice or "").strip()
    if not voice:
        raise HTTPException(status_code=400, detail="Voice is required")

    rate_edge = _rate_to_edge(req.rate)
    pitch = "+0Hz"

    # Cache by content+params to avoid re-generating repeatedly
    vid = (req.video_id or "global").strip() or "global"
    key = f"{voice}|{rate_edge}|{pitch}|{text}".encode("utf-8", errors="ignore")
    h = hashlib.sha1(key).hexdigest()[:16]

    tts_dir = os.path.join(settings.OUTPUT_DIR, "tts", vid)
    os.makedirs(tts_dir, exist_ok=True)
    filename = f"tts_{h}.mp3"
    out_path = os.path.join(tts_dir, filename)

    def _ffmpeg_atempo_args(rate: float) -> list[str]:
        r = max(0.5, min(2.0, float(rate)))
        # atempo accepts 0.5..2.0; chain if needed (we clamp, so one is fine)
        return ["-filter:a", f"atempo={r:.3f}"]

    if not os.path.exists(out_path):
        if voice.startswith("gtts:"):
            try:
                from gtts import gTTS
            except Exception:
                raise HTTPException(
                    status_code=503,
                    detail="TTS backend not installed. Install 'gTTS' (and optionally 'edge-tts') to enable Read Aloud.",
                )

            parts = voice.split(":")
            lang = parts[1] if len(parts) > 1 and parts[1] else "en"
            tld = parts[2] if len(parts) > 2 and parts[2] else "com"

            tmp_path = os.path.join(tts_dir, f"tmp_{h}.mp3")
            try:
                tts = gTTS(text=text, lang=lang, tld=tld)
                tts.save(tmp_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"TTS generation failed (gTTS): {str(e)}")

            # Apply speed if possible via ffmpeg; otherwise, use raw audio.
            if abs(float(req.rate) - 1.0) > 0.01:
                try:
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        tmp_path,
                        *_ffmpeg_atempo_args(float(req.rate)),
                        out_path,
                    ]
                    subprocess.run(cmd, check=True)
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                except Exception:
                    # If ffmpeg isn't available, fall back to the unmodified mp3.
                    os.replace(tmp_path, out_path)
            else:
                os.replace(tmp_path, out_path)

        else:
            try:
                import edge_tts
            except Exception:
                raise HTTPException(
                    status_code=503,
                    detail="TTS backend not installed. Install 'edge-tts' (or pick a Google TTS voice) to enable Read Aloud.",
                )

            communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate_edge, pitch=pitch)
            try:
                await communicate.save(out_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    return TtsResponse(
        audio_url=f"/api/tts/audio/{vid}/{filename}",
        voice=voice,
        rate=float(req.rate),
    )


@router.get("/tts/audio/{video_id}/{filename}")
async def get_tts_audio(video_id: str, filename: str):
    """Serve a generated TTS MP3 from disk."""
    safe_vid = (video_id or "").replace("..", "").replace("/", "_").replace("\\", "_")
    safe_fn = (filename or "").replace("..", "").replace("/", "_").replace("\\", "_")

    path = os.path.join(settings.OUTPUT_DIR, "tts", safe_vid, safe_fn)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="TTS audio not found")
    return FileResponse(path, media_type="audio/mpeg", filename=safe_fn)
