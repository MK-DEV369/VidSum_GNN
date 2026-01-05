from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import shutil
import os
import uuid
import json

from vidsum_gnn.db.client import get_db
from vidsum_gnn.db.models import Video, Summary, Shot
from vidsum_gnn.core.config import settings
from vidsum_gnn.api.tasks import process_video_task
from vidsum_gnn.utils.logging import get_logger
from pydantic import BaseModel

router = APIRouter()
logger = get_logger(__name__)

class ProcessRequest(BaseModel):
    target_duration: int = 60
    selection_method: str = "greedy"

async def broadcast_log(request: Request, video_id: str, log_data: dict):
    """Broadcast log to connected WebSocket clients"""
    try:
        from vidsum_gnn.api.main import manager
        await manager.send_log(video_id, log_data)
    except Exception as e:
        logger.error(f"Failed to broadcast log: {e}")

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
    """Upload and process a video file"""
    video_id = str(uuid.uuid4())
    filename = f"{video_id}_{file.filename}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    
    # Log upload start
    await broadcast_log(request, video_id, {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "level": "INFO",
        "message": f"Starting upload: {file.filename} ({file.size or 0} bytes)",
        "stage": "UPLOAD"
    })
    
    try:
        # Save file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to {filepath}")
        
        # Create DB entry
        video = Video(
            video_id=video_id,
            filename=filename,
            status="processing",
            target_duration=60,  # Default for video if generated
            selection_method="greedy"
        )
        db.add(video)
        await db.commit()
        
        # Log upload complete
        await broadcast_log(request, video_id, {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "level": "SUCCESS",
            "message": f"Upload complete. Starting processing ({summary_type}, {text_length} text, {'with' if generate_video else 'no'} video)",
            "stage": "UPLOAD",
            "progress": 15
        })
        
        # Start background processing
        if background_tasks:
            background_tasks.add_task(
                process_video_task,
                video_id,
                {
                    "text_length": text_length,
                    "summary_format": summary_format,
                    "summary_type": summary_type,
                    "generate_video": generate_video,
                    "target_duration": 60,  # Default for video generation
                    "selection_method": "greedy"
                }
            )
        
        return {
            "video_id": video_id,
            "status": "queued",
            "message": "Video uploaded successfully. Processing started."
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Upload error: {error_msg}")
        
        await broadcast_log(request, video_id, {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "level": "ERROR",
            "message": f"Upload failed: {error_msg}",
            "stage": "UPLOAD"
        })
        
        raise HTTPException(status_code=400, detail=error_msg)

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
    
    return {"message": "Processing started", "task_id": video_id}

@router.get("/status/{video_id}")
async def get_status(video_id: str, db: AsyncSession = Depends(get_db)):
    """Get processing status of a video"""
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return {
        "video_id": video_id,
        "status": video.status,
        "target_duration": video.target_duration,
        "selection_method": video.selection_method
    }

@router.get("/results/{video_id}")
async def get_results(video_id: str, db: AsyncSession = Depends(get_db)):
    """Get summary results for a video"""
    result = await db.execute(select(Summary).where(Summary.video_id == video_id))
    summaries = result.scalars().all()
    return summaries

@router.get("/shot-scores/{video_id}")
async def get_shot_scores(video_id: str, db: AsyncSession = Depends(get_db)):
    """Get shot-level importance scores"""
    result = await db.execute(select(Shot).where(Shot.video_id == video_id))
    shots = result.scalars().all()
    return shots

@router.get("/videos")
async def list_videos(db: AsyncSession = Depends(get_db)):
    """List all uploaded videos"""
    result = await db.execute(select(Video))
    videos = result.scalars().all()
    return videos

@router.get("/summary/{video_id}/text")
async def get_text_summary(video_id: str, format: str = "bullet", db: AsyncSession = Depends(get_db)):
    """Get text summary for a video in specified format"""
    result = await db.execute(
        select(Summary).where(Summary.video_id == video_id)
    )
    summary = result.scalar_one_or_none()
    
    if not summary:
        raise HTTPException(status_code=404, detail="Text summary not found")
    
    # Return appropriate format
    summary_map = {
        "bullet": summary.text_summary_bullet,
        "structured": summary.text_summary_structured,
        "plain": summary.text_summary_plain
    }
    
    return {
        "video_id": video_id,
        "summary": summary_map.get(format, summary.text_summary_bullet),
        "format": format,
        "style": summary.summary_style
    }

@router.get("/download/{video_id}")
async def download_summary_video(video_id: str, db: AsyncSession = Depends(get_db)):
    """Download the generated summary video"""
    result = await db.execute(
        select(Summary).where(Summary.video_id == video_id)
    )
    summary = result.scalar_one_or_none()
    
    if not summary or not summary.video_path:
        raise HTTPException(status_code=404, detail="Summary video not found")
    
    from pathlib import Path
    video_path = Path(summary.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Summary video file not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"summary_{video_id}.mp4"
    )
