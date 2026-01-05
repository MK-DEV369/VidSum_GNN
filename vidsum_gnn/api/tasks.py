import asyncio
import os
import torch
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from vidsum_gnn.db.client import AsyncSessionLocal
from vidsum_gnn.db.models import Video, Shot, Summary
from vidsum_gnn.processing.video import probe_video, transcode_video
from vidsum_gnn.processing.shot_detection import detect_shots, sample_frames_for_shots
from vidsum_gnn.processing.audio import extract_audio_segment
from vidsum_gnn.features.visual import VisualEncoder
from vidsum_gnn.features.audio import AudioEncoder
from vidsum_gnn.graph.builder import GraphBuilder
from vidsum_gnn.graph.model import VidSumGNN
from vidsum_gnn.core.config import settings
from vidsum_gnn.utils.logging import get_logger
from vidsum_gnn.inference.service import get_inference_service  # NEW: Use refactored service

logger = get_logger(__name__)


async def _send_log(video_id: str, message: str, level: str = "INFO", stage: str | None = None, progress: int | None = None):
    from vidsum_gnn.api.main import manager  # Local import to avoid circular dependency
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

async def process_video_task(video_id: str, config: dict):
    """
    Main pipeline orchestration.
    """
    async with AsyncSessionLocal() as db:
        try:
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
            
            # 3. Shot Detection
            video.status = "shot_detection"
            await db.commit()
            await _send_log(video_id, "Running shot detection", stage="shot_detection", progress=35)
            
            shots_times = await detect_shots(canonical_path)
            await _send_log(video_id, f"Detected {len(shots_times)} shots", stage="shot_detection", progress=45)
            
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
            
            # Extract keyframes
            keyframe_paths = await sample_frames_for_shots(canonical_path, shots_times, video_id)
            
            # Extract audio for shots (optional, can be slow for many shots)
            # For prototype, let's extract full audio and process chunks in memory or 
            # just extract a few for demo. 
            # Let's extract per shot as per spec.
            audio_paths = []
            audio_dir = os.path.join(settings.PROCESSED_DIR, video_id, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            
            # Extract audio segments from video
            for i, (start, end) in enumerate(shots_times):
                path = os.path.join(audio_dir, f"shot_{i:04d}.mp3")
                await extract_audio_segment(canonical_path, start, end, path)
                audio_paths.append(path)

            # Encoders
            vis_encoder = VisualEncoder()
            aud_encoder = AudioEncoder() # This will fail on dummy files, so we need real files or mock
            
            # Extract visual and audio features
            vis_feats = vis_encoder.encode(keyframe_paths)
            aud_feats = aud_encoder.encode(audio_paths)
            
            # Fuse: Concat
            features = torch.cat([vis_feats, aud_feats], dim=1) # (N, 1536)
            
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
            
            await _send_log(video_id, "Generating importance scores and text summary", stage="gnn_inference", progress=75)
            
            # Run full pipeline: GNN + Text Summarization
            gnn_scores, text_summary = inference_service.process_video_pipeline(
                node_features=graph_data.x,
                edge_index=graph_data.edge_index,
                audio_paths=audio_paths,
                summary_type=summary_type,
                text_length=text_length,
                summary_format=summary_format
            )
            
            # Convert tensor scores to list
            scores = gnn_scores.squeeze().tolist()
            if isinstance(scores, float):
                scores = [scores]
            
            # Store importance scores in shot records
            for i, shot_data in enumerate(shots_data):
                shot = Shot(**shot_data)
                if i < len(scores):
                    shot.importance_score = scores[i]
                db.add(shot)
            await db.commit()
                
            # 6. Text Summary Generation
            await _send_log(video_id, "Generating text summaries", stage="assembling", progress=80)
            
            # 7. Finalize
            video.status = "completed"
            await _send_log(video_id, "Processing complete", stage="completed", progress=95)
            
            # Save summary record with text summary in all three formats
            # (Generate all formats regardless of user choice for future retrieval)
            summary_text = text_summary  # This is the selected format
            
            # Generate other formats for storage
            all_formats = {}
            for fmt in ["bullet", "structured", "plain"]:
                _, fmt_summary = inference_service.process_video_pipeline(
                    node_features=graph_data.x,
                    edge_index=graph_data.edge_index,
                    audio_paths=audio_paths,
                    summary_type=summary_type,
                    text_length=text_length,
                    summary_format=fmt
                )
                all_formats[fmt] = fmt_summary
            
            summary = Summary(
                summary_id=f"sum_{video_id}",
                video_id=video_id,
                type="text_only",
                duration=0,
                video_path=None,
                text_summary_bullet=all_formats["bullet"],
                text_summary_structured=all_formats["structured"],
                text_summary_plain=all_formats["plain"],
                summary_style=summary_type,
                config_json=config
            )
            db.add(summary)
            await db.commit()
            
            # Log text summary preview
            summary_preview = text_summary[:200] if text_summary else "No summary"
            await _send_log(video_id, f"Text summary generated: {summary_preview}...", stage="completed", progress=98)
            await _send_log(video_id, "Processing complete", level="SUCCESS", stage="completed", progress=100)
            
        except Exception as e:
            logger.error(f"Task failed for video {video_id}: {e}", exc_info=True)
            print(f"Task failed: {e}")
            video.status = "failed"
            await db.commit()
            await _send_log(video_id, f"Processing failed: {e}", level="ERROR", stage="failed", progress=100)
