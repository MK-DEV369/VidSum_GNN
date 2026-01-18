from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any
import shutil
import os
import uuid
import json
from datetime import datetime

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

class SummaryResultResponse(BaseModel):
    """Complete summary result"""
    video_id: str
    status: str
    text_summaries: Dict[str, str]
    summary_type: str
    fallback_used: bool
    generated_at: Optional[str] = None

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
    text_length: str = "medium",
    summary_format: str = "bullet",
    summary_type: str = "balanced",
    generate_video: bool = False,
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
        generated_at=summary.generated_at.isoformat() if summary.generated_at else None
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
    
    return TextSummaryResponse(
        video_id=video_id,
        summary=requested_summary,
        format=format,
        style=summary.summary_style,
        generated_at=summary.generated_at.isoformat() if summary.generated_at else None
    )

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
