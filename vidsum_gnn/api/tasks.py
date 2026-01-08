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
from vidsum_gnn.inference.gemini_fallback import get_gemini_summarizer  # NEW: Gemini fallback

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
            
            await _send_log(video_id, "Generating binary predictions and text summary", stage="gnn_inference", progress=75)
            
            # Run full pipeline: GNN + Text Summarization
            try:
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
                
                await _send_log(video_id, "✓ GNN inference successful", stage="gnn_inference", progress=80)
                used_fallback = False
                
            except Exception as gnn_error:
                logger.error(f"GNN processing failed for {video_id}: {gnn_error}", exc_info=True)
                await _send_log(video_id, f"GNN inference failed: {gnn_error}. Attempting Gemini fallback...", level="WARNING", stage="gnn_inference", progress=75)
                
                # Fallback to Gemini API for summarization
                gemini = get_gemini_summarizer()
                if gemini.is_available():
                    await _send_log(video_id, "⚠️  Switching to Gemini API fallback for text summarization", level="WARNING", stage="gnn_fallback", progress=76)
                    text_summary, gemini_metadata = gemini.summarize_video_from_path(
                        video_path=canonical_path,
                        summary_type=summary_type,
                        text_length=text_length,
                        summary_format=summary_format
                    )
                    
                    if text_summary:
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
            
            # 6a. Transcribe audio from selected shots
            logger.info(f"Transcribing {len(audio_paths)} audio files")
            transcriber = inference_service.manager.get_whisper()
            transcripts = []
            
            for i, audio_path in enumerate(audio_paths):
                audio_path_obj = audio_path if isinstance(audio_path, str) else str(audio_path)
                if os.path.exists(audio_path_obj):
                    try:
                        transcript = transcriber.transcribe(audio_path_obj, cache_dir=None)
                        transcripts.append(transcript if transcript else "")
                        logger.info(f"✓ Transcribed shot {i}: {len(transcript)} chars")
                    except Exception as e:
                        logger.error(f"Transcription failed for shot {i}: {e}")
                        transcripts.append("")
                else:
                    logger.warning(f"Audio file not found for shot {i}: {audio_path_obj}")
                    transcripts.append("")
            
            logger.info(f"Transcription complete ({sum(1 for t in transcripts if t)} successful)")
            
            # 6b. Calculate top-K shots to include in summary
            n = len(gnn_scores) if not used_fallback else len(scores)
            k_ratio = max(0.0, min(1.0, float(getattr(settings, "TOPK_RATIO", 0.15))))
            top_k = max(1, int(__import__('numpy').ceil(k_ratio * n))) if n > 0 else 1
            logger.info(f"Using top {top_k} shots out of {n} total ({k_ratio*100:.0f}% ratio)")
            
            # 6c. Generate the summarizer
            summarizer = inference_service.manager.get_summarizer()
            
            # 6d. Generate summaries in all three formats using the same GNN scores
            all_formats = {}
            for fmt in ["bullet", "structured", "plain"]:
                logger.info(f"Generating {fmt} summary for {video_id}")
                fmt_summary = summarizer.summarize(
                    transcripts=transcripts,
                    gnn_scores=gnn_scores.tolist() if not used_fallback else scores,
                    summary_type=summary_type,
                    text_length=text_length,
                    summary_format=fmt,
                    top_k=top_k
                )
                all_formats[fmt] = fmt_summary
                logger.info(f"✓ Generated {fmt} summary ({len(fmt_summary)} chars)")
            
            await _send_log(video_id, "Summary generation complete", stage="summarization", progress=90)
            
            # 7. Finalize
            video.status = "completed"
            await _send_log(video_id, "Processing complete", stage="completed", progress=95)
            
            # Save summary record with text summaries in all three formats
            fallback_note = " [GEMINI FALLBACK]" if used_fallback else ""
            text_summary_preview = all_formats.get("bullet", "")[:150] if all_formats.get("bullet") else "No summary"
            
            # Log text summary preview
            await _send_log(
                video_id,
                f"✓ Text summary generated{fallback_note}: {text_summary_preview}...",
                level="SUCCESS",
                stage="completed",
                progress=98
            )
            
            # Create and save summary record
            summary = Summary(
                summary_id=f"sum_{video_id}",
                video_id=video_id,
                type="text_only",
                duration=0,
                video_path=None,
                text_summary_bullet=all_formats.get("bullet", ""),
                text_summary_structured=all_formats.get("structured", ""),
                text_summary_plain=all_formats.get("plain", ""),
                summary_style=summary_type,
                config_json={
                    **config,
                    "fallback_used": used_fallback,
                    "text_length": text_length,
                    "summary_type": summary_type,
                    "generated_formats": list(all_formats.keys())
                }
            )
            db.add(summary)
            await db.commit()
            
            await _send_log(
                video_id,
                "✓ Processing complete",
                level="SUCCESS",
                stage="completed",
                progress=100
            )
            
        except Exception as e:
            logger.error(f"Task failed for video {video_id}: {e}", exc_info=True)
            print(f"Task failed: {e}")
            video.status = "failed"
            await db.commit()
            await _send_log(video_id, f"Processing failed: {e}", level="ERROR", stage="failed", progress=100)
