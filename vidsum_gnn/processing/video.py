import os
import json
import subprocess
import asyncio
import gc
from typing import Dict, Any, List, Tuple
from vidsum_gnn.core.config import settings
from vidsum_gnn.utils import get_logger, PipelineStage

# Memory management
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

def clear_memory():
    """Clear Python and GPU memory."""
    gc.collect()
    if TORCH_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

async def probe_video(file_path: str) -> Dict[str, Any]:
    """
    Probe video file using ffprobe to get metadata.
    """
    logger = get_logger(__name__)
    logger.info(f"Probing video: {file_path}")
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        logger.error(f"ffprobe failed: {stderr.decode()}")
        raise RuntimeError(f"ffprobe failed: {stderr.decode()}")
        
    data = json.loads(stdout.decode())
    
    # Extract useful info
    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if not video_stream:
        logger.error("No video stream found in file")
        raise ValueError("No video stream found")
        
    duration = float(data["format"].get("duration", 0))
    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = map(int, fps_str.split('/'))
    fps = num / den if den != 0 else 30.0
    
    metadata = {
        "duration": duration,
        "fps": fps,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_name", "unknown")
    }
    
    logger.info(f"Video metadata: {duration:.1f}s, {fps:.1f}fps, {metadata['width']}x{metadata['height']}")
    return metadata

async def transcode_video(input_path: str, output_path: str = None) -> str:
    """
    Transcode video to canonical H.264/MP4 if needed.
    If output_path is None, creates one in TEMP_DIR.
    """
    logger = get_logger(__name__)
    logger.info(f"Transcoding video: {input_path}", stage=PipelineStage.UPLOAD)

    if output_path is None:
        filename = os.path.basename(input_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(settings.TEMP_DIR, f"{name}_canonical.mp4")

    # Check if already exists
    if os.path.exists(output_path):
        logger.info(f"Using existing transcoded file: {output_path}")
        return output_path

    # Try libopenh264 first (available in your build), then fall back to mpeg4 or copy
    primary_cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libopenh264",
        "-b:v", "2M",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",
        output_path
    ]

    fallback_cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "mpeg4",
        "-qscale:v", "2",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",
        output_path
    ]
    
    # If input is already h264, just copy it
    copy_cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        "-y",
        output_path
    ]

    async def _run(cmd, timeout=300):  # 5 minute timeout
        logger.debug(f"Transcoding command: {' '.join(cmd)}")
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return process.returncode, stdout, stderr
        except asyncio.TimeoutError:
            logger.error(f"FFmpeg transcoding timed out after {timeout}s")
            process.kill()
            await process.wait()
            raise RuntimeError(f"ffmpeg transcoding timed out after {timeout}s")

    # First check if input is already h264 and try copying
    try:
        metadata = await probe_video(input_path)
        if metadata.get('codec') == 'h264':
            logger.info("Input is already H.264, attempting stream copy (fastest)")
            code, _, stderr = await _run(copy_cmd)
            if code == 0:
                logger.info("Stream copy successful")
                clear_memory()
                return output_path
            else:
                logger.info("Stream copy failed, trying re-encode")
    except Exception as e:
        logger.warning(f"Could not probe video codec: {e}")
    
    # Try primary codec
    code, _, stderr = await _run(primary_cmd)
    if code != 0:
        err_text = stderr.decode()
        logger.warning(f"FFmpeg primary codec (libopenh264) failed, retrying with mpeg4")
        logger.debug(f"Primary codec error: {err_text[:500]}")  # Log first 500 chars only
        
        code, _, stderr = await _run(fallback_cmd)
        if code != 0:
            logger.error(f"FFmpeg transcoding failed after fallback: {stderr.decode()[:500]}")
            raise RuntimeError(f"ffmpeg transcoding failed with both codecs")

    logger.info(f"Transcoding complete: {output_path}")
    clear_memory()  # Clear memory after transcoding
    return output_path

def get_chunks(duration: float, chunk_size: int = 300, overlap: int = 30) -> List[Tuple[float, float]]:
    """
    Generate start/end times for chunks.
    """
    logger = get_logger(__name__)
    chunks = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_size, duration)
        chunks.append((start, end))
        if end == duration:
            break
        start += (chunk_size - overlap)
    
    logger.info(f"Generated {len(chunks)} chunks for {duration:.1f}s video (chunk_size={chunk_size}s, overlap={overlap}s)")
    return chunks

async def merge_important_shots(
    input_video: str,
    shots_times: List[Tuple[float, float]],
    importance_scores: List[float],
    threshold: float = 0.5,
    output_path: str = None,
    max_duration: int = 600
) -> str:
    """
    Create a merged video containing only the important shots.
    
    Args:
        input_video: Path to the original video
        shots_times: List of (start_sec, end_sec) tuples for all shots
        importance_scores: GNN importance scores for each shot (0-1)
        threshold: Importance threshold (default 0.5) - shots with score >= threshold are included
        output_path: Where to save the merged video (auto-generated if None)
        max_duration: Maximum duration for the merged video in seconds
        
    Returns:
        Path to the merged video file
    """
    logger = get_logger(__name__)
    logger.info(f"Merging important shots (threshold={threshold})")
    
    # Filter shots by importance
    important_shots = [
        (shots_times[i], importance_scores[i])
        for i in range(len(shots_times))
        if i < len(importance_scores) and importance_scores[i] >= threshold
    ]
    
    if not important_shots:
        # If no shots meet threshold, use top 20% of shots (min 10, max 50)
        num_fallback = max(10, min(50, len(shots_times) // 5))
        logger.warning(f"No shots met threshold {threshold}, using top {num_fallback} shots")
        sorted_shots = sorted(
            zip(shots_times, importance_scores),
            key=lambda x: x[1],
            reverse=True
        )
        important_shots = sorted_shots[:num_fallback]
    
    logger.info(f"Selected {len(important_shots)} important shots out of {len(shots_times)}")
    
    if output_path is None:
        video_id = os.path.basename(input_video).split('_')[0]
        output_path = os.path.join(settings.TEMP_DIR, f"{video_id}_important_shots.mp4")
    
    # Create concat demux file for ffmpeg
    concat_file = output_path.replace('.mp4', '_concat.txt')
    
    try:
        # Calculate total duration of important shots
        total_duration = sum(end - start for (start, end), _ in important_shots)
        
        # If total duration exceeds max, sample shots to reduce it
        if total_duration > max_duration:
            logger.info(f"Important shots total {total_duration:.1f}s exceeds max {max_duration}s, sampling...")
            # Keep only top shots until we reach max_duration
            sampled_shots = []
            current_duration = 0
            for shot, score in sorted(important_shots, key=lambda x: x[1], reverse=True):
                shot_duration = shot[1] - shot[0]
                if current_duration + shot_duration <= max_duration:
                    sampled_shots.append((shot, score))
                    current_duration += shot_duration
            
            important_shots = sorted(sampled_shots, key=lambda x: x[0][0])  # Re-sort by time
            total_duration = current_duration
        
        logger.info(f"Creating merged video with {len(important_shots)} shots ({total_duration:.1f}s total)")
        
        # Extract individual segments with video+audio
        segments = []
        for i, ((start, end), score) in enumerate(important_shots):
            segment_path = output_path.replace('.mp4', f'_seg_{i:03d}.mp4')
            
            seg_cmd = [
                "ffmpeg",
                "-i", input_video,
                "-ss", str(start),
                "-to", str(end),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                "-y",
                segment_path
            ]
            
            logger.debug(f"Extracting segment {i+1}/{len(important_shots)}: {start:.1f}s-{end:.1f}s")
            
            process = await asyncio.create_subprocess_exec(
                *seg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            
            if process.returncode == 0:
                segments.append(segment_path)
                logger.debug(f"✓ Extracted segment {i+1}")
            else:
                logger.error(f"Failed to extract segment {i}: {stderr.decode()[:200]}")
        
        if not segments:
            logger.error("No segments extracted successfully")
            raise RuntimeError("Failed to extract any video segments")
        
        # Concatenate all segments
        if len(segments) == 1:
            # Single segment, just rename it
            import shutil
            shutil.move(segments[0], output_path)
            logger.info(f"✓ Single segment, moved to {output_path}")
        else:
            # Multiple segments, use concat demuxer
            concat_file_path = output_path.replace('.mp4', '_concat.txt')
            with open(concat_file_path, 'w') as f:
                for seg in segments:
                    f.write(f"file '{seg}'\n")
            
            logger.debug(f"Concatenating {len(segments)} segments")
            
            concat_cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file_path,
                "-map", "0:v:0",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                "-y",
                output_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *concat_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)
            
            if process.returncode != 0:
                logger.error(f"FFmpeg concat failed: {stderr.decode()[:300]}")
                raise RuntimeError(f"Failed to concatenate segments: {stderr.decode()[:300]}")
            
            logger.debug(f"✓ Concatenated {len(segments)} segments")
            
            # Cleanup segment files
            for seg in segments:
                try:
                    os.remove(seg)
                except Exception as e:
                    logger.warning(f"Failed to cleanup segment {seg}: {e}")
            
            try:
                os.remove(concat_file_path)
            except Exception:
                pass
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024*1024)
            logger.info(f"✓ Merged video created: {output_path} ({file_size:.1f}MB)")
            return output_path
        else:
            raise RuntimeError(f"Output file not created: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in merge_important_shots: {str(e)[:200]}")
        if os.path.exists(concat_file):
            try:
                os.remove(concat_file)
            except:
                pass
        raise