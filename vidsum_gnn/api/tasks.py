import asyncio
import os
import torch
import shutil
import time
from datetime import datetime, timedelta
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

logger = get_logger(__name__)

# Batch processing constants
CHUNK_DURATION = 900  # 15 minutes per chunk for large videos
MAX_VIDEO_DURATION = 3600  # 1 hour - if video > 1 hour, use batch processing

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
    """Process a single chunk of video and return scores"""
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
        logger.error(f"Failed to extract chunk {chunk_index}: {stderr.decode()[:200]}")
        return None, None
    
    # Run feature extraction and GNN on chunk
    try:
        shots_times = await detect_shots(chunk_path)
        keyframe_paths = await sample_frames_for_shots(chunk_path, shots_times, f"{video_id}_chunk_{chunk_index}")
        
        audio_dir = os.path.join(settings.PROCESSED_DIR, f"{video_id}_chunk_{chunk_index}", "audio")
        os.makedirs(audio_dir, exist_ok=True)
        audio_paths = []
        
        for i, (start, end) in enumerate(shots_times):
            path = os.path.join(audio_dir, f"shot_{i:04d}.mp3")
            await extract_audio_segment(chunk_path, start, end, path)
            audio_paths.append(path)
        
        vis_encoder = VisualEncoder()
        aud_encoder = AudioEncoder()
        
        vis_feats = vis_encoder.encode(keyframe_paths)  # (N, 768)
        aud_feats = aud_encoder.encode(audio_paths)     # (N, 768)
        
        # Clear CUDA memory after feature extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Extract handcrafted features (duration, position, etc.)
        handcrafted_feats = extract_handcrafted_features(
            shots_times=shots_times,
            video_duration=video_duration_seconds
        )  # (N, 14)
        
        # Generate text embeddings from transcriptions using existing service
        transcription_service = TranscriptionService(device="cpu")  # Force CPU to save GPU memory
        text_embedder = TextEmbedder(device="cpu")  # Force CPU for text embedder too
        
        transcriptions = []
        for audio_path in audio_paths:
            try:
                # Use existing transcription service (already handles Whisper)
                text = transcription_service.transcribe_audio(audio_path)
                transcriptions.append(text if text else "")
            except Exception as e:
                logger.warning(f"Failed to transcribe {audio_path}: {e}")
                transcriptions.append("")  # Empty text will get zero embedding
        
        text_feats_np = text_embedder.batch_encode(transcriptions)  # (N, 384)
        text_feats = torch.from_numpy(text_feats_np).float() if len(text_feats_np) > 0 else torch.zeros(len(transcriptions), 384)
        
        # Clear memory after text processing
        del transcription_service
        del text_embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        gnn_scores, _ = inference_service.predict_importance_scores(
            graph_data.x,
            graph_data.edge_index
        )
        
        # Convert to list and adjust times to absolute video time
        scores = gnn_scores.squeeze().tolist() if isinstance(gnn_scores, torch.Tensor) else gnn_scores
        if isinstance(scores, float):
            scores = [scores]
        
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
        
        return adjusted_shots, scores
        
    except Exception as e:
        logger.error(f"Chunk {chunk_index} processing failed: {e}")
        return None, None


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
            
            # Probe video for metadata
            video_metadata = await probe_video(canonical_path)
            video_duration_seconds = video_metadata['duration']
            logger.info(f"Video duration: {video_duration_seconds:.1f}s")
            await _send_log(video_id, f"Video duration: {video_duration_seconds:.1f}s", stage="preprocessing", progress=32)
            
            # 3. Shot Detection (use lower threshold 0.25 to detect more shots)
            video.status = "shot_detection"
            await db.commit()
            await _send_log(video_id, "Running shot detection", stage="shot_detection", progress=35)
            
            shots_times = await detect_shots(canonical_path, threshold=0.25)  # Lower threshold for more shots
            
            # Filter out very short shots (< 1 second) which are usually UI noise/transitions
            min_shot_duration = 1.0
            original_count = len(shots_times)
            shots_times = [(s, e) for s, e in shots_times if (e - s) >= min_shot_duration]
            
            if len(shots_times) < original_count:
                logger.info(f"Filtered out {original_count - len(shots_times)} short shots (<{min_shot_duration}s)")
            
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
            aud_encoder = AudioEncoder()
            
            # Extract visual and audio features in batches to avoid CUDA OOM
            logger.info(f"Extracting features for {len(keyframe_paths)} shots")
            batch_size = 2  # Reduced from 4 - audio files need more memory on 8GB GPU
            vis_feats_list = []
            aud_feats_list = []
            
            for batch_idx in range(0, len(keyframe_paths), batch_size):
                batch_end = min(batch_idx + batch_size, len(keyframe_paths))
                batch_num = (batch_idx // batch_size) + 1
                total_batches = (len(keyframe_paths) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({batch_idx}-{batch_end}/{len(keyframe_paths)})")
                
                # Process visual features for this batch
                vis_batch = vis_encoder.encode(keyframe_paths[batch_idx:batch_end])
                vis_feats_list.append(vis_batch)
                
                # Process audio features for this batch
                aud_batch = aud_encoder.encode(audio_paths[batch_idx:batch_end])
                aud_feats_list.append(aud_batch)
                
                # Clear CUDA cache between batches more aggressively
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Force garbage collection
                    import gc
                    gc.collect()
            
            # Concatenate all batches
            vis_feats = torch.cat(vis_feats_list, dim=0) if vis_feats_list else torch.empty(0, 768)
            aud_feats = torch.cat(aud_feats_list, dim=0) if aud_feats_list else torch.empty(0, 768)
            
            # Clear CUDA memory after concatenating features
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
            
            # 6d. Generate ONLY the requested format
            logger.info(f"Generating {summary_format} summary for {video_id} (format: {summary_format}, length: {text_length}, type: {summary_type})")
            text_summary = summarizer.summarize(
                transcripts=transcripts,
                gnn_scores=gnn_scores.tolist() if not used_fallback else scores,
                summary_type=summary_type,
                text_length=text_length,
                summary_format=summary_format,
                top_k=top_k
            )
            all_formats = {summary_format: text_summary}
            logger.info(f"✓ Generated {summary_format} summary ({len(text_summary)} chars)")
            
            await _send_log(video_id, "Summary generation complete", stage="summarization", progress=90)
            
            # 7. Generate merged video of important shots
            await _send_log(video_id, "Creating merged video of important shots", stage="video_merge", progress=92)
            try:
                # Use adaptive threshold based on score distribution
                scores_array = gnn_scores if not used_fallback else scores
                if isinstance(scores_array, torch.Tensor):
                    scores_array = scores_array.cpu().numpy()
                else:
                    scores_array = __import__('numpy').array(scores_array)
                
                # Calculate threshold as 30th percentile to select top 70% of shots
                # This gives more shots for a richer compilation (10min video = ~50-70 shots)
                import numpy as np
                if len(scores_array) > 0:
                    threshold = float(np.percentile(scores_array, 30)) if len(scores_array) > 1 else 0.3
                else:
                    threshold = 0.3
                
                logger.info(f"Adaptive threshold (30th percentile): {threshold:.3f}, will select ~{int(len(scores_array) * 0.7)} shots")
                
                merged_video_path = await merge_important_shots(
                    input_video=canonical_path,
                    shots_times=shots_times,
                    importance_scores=scores_array.tolist() if hasattr(scores_array, 'tolist') else scores_array,
                    threshold=threshold,
                    max_duration=300  # Max 5 minutes for merged video
                )
                
                await _send_log(video_id, f"✓ Merged video created: {os.path.basename(merged_video_path)}", stage="video_merge", progress=93)
                logger.info(f"Merged video: {merged_video_path}")
            except Exception as e:
                merged_video_path = None
                logger.warning(f"Failed to create merged video: {e}")
                await _send_log(video_id, f"⚠️  Merged video skipped: {str(e)[:100]}", level="WARNING", stage="video_merge", progress=93)
            
            # 8. Finalize
            video.status = "completed"
            await _send_log(video_id, "Processing complete", stage="completed", progress=95)
            
            # Save summary record with only the requested format
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
            
            # Create and save summary record - only store requested format
            summary_kwargs = {
                "summary_id": f"sum_{video_id}",
                "video_id": video_id,
                "type": "text_only",
                "duration": 0,
                "video_path": merged_video_path,  # Store merged video path
                "summary_style": summary_type,
                "config_json": {
                    **config,
                    "fallback_used": used_fallback,
                    "text_length": text_length,
                    "summary_type": summary_type,
                    "generated_formats": [summary_format],
                    "merged_video_enabled": merged_video_path is not None
                }
            }
            
            # Store only the requested format
            if summary_format == "bullet":
                summary_kwargs["text_summary_bullet"] = text_summary
            elif summary_format == "structured":
                summary_kwargs["text_summary_structured"] = text_summary
            elif summary_format == "plain":
                summary_kwargs["text_summary_plain"] = text_summary
            
            summary = Summary(**summary_kwargs)
            db.add(summary)
            await db.commit()
            
            await _send_log(
                video_id,
                "✓ Processing complete",
                level="SUCCESS",
                stage="completed",
                progress=100
            )
            
            # 9. Cleanup temporary files
            await _send_log(video_id, "Cleaning up temporary files...", stage="cleanup", progress=99)
            try:
                # Clean upload directory for this video
                uploaded_file_path = os.path.join(settings.UPLOAD_DIR, video.filename)
                if os.path.exists(uploaded_file_path):
                    os.remove(uploaded_file_path)
                    logger.info(f"Cleaned up uploaded file: {uploaded_file_path}")
                
                # Clean processed directory for this video (frames, audio segments, etc.)
                video_processed_dir = os.path.join(settings.PROCESSED_DIR, video_id)
                if os.path.exists(video_processed_dir):
                    shutil.rmtree(video_processed_dir)
                    logger.info(f"Cleaned up processed directory: {video_processed_dir}")
                
                # Clean canonical video if it's different from source
                if canonical_path != uploaded_file_path and os.path.exists(canonical_path):
                    os.remove(canonical_path)
                    logger.info(f"Cleaned up canonical video: {canonical_path}")
                
                await _send_log(video_id, "✓ Temporary files cleaned up", stage="cleanup", progress=100)
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error for {video_id}: {cleanup_error}")
                await _send_log(video_id, f"⚠️  Partial cleanup: {str(cleanup_error)[:100]}", level="WARNING", stage="cleanup", progress=100)

            
        except Exception as e:
            logger.error(f"Task failed for video {video_id}: {e}", exc_info=True)
            print(f"Task failed: {e}")
            video.status = "failed"
            await db.commit()
            await _send_log(video_id, f"Processing failed: {e}", level="ERROR", stage="failed", progress=100)
