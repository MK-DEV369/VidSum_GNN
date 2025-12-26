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

    primary_cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-b:v", "3M",
        "-maxrate", "3M",
        "-bufsize", "6M",
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

    code, _, stderr = await _run(primary_cmd)
    if code != 0:
        err_text = stderr.decode()
        logger.warning(f"FFmpeg primary codec failed, retrying with mpeg4. Error: {err_text}")
        logger.info("Starting FFmpeg transcoding with fallback codec (mpeg4)")
        # Ensure fallback sets sane pixel format
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
        code, _, stderr = await _run(fallback_cmd)
        if code != 0:
            logger.error(f"FFmpeg transcoding failed after fallback: {stderr.decode()}")
            raise RuntimeError(f"ffmpeg transcoding failed: {stderr.decode()}")

    logger.info(f"Transcoding complete: {output_path}")
        # Detect available encoders
        encoders_cmd = ["ffmpeg", "-hide_banner", "-encoders"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *encoders_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            out, _ = await proc.communicate()
            encoders_txt = out.decode()
        except Exception:
            encoders_txt = ""

        use_openh264 = "libopenh264" in encoders_txt
        use_libx264 = "libx264" in encoders_txt

        if use_openh264:
            logger.info("Starting FFmpeg transcoding with primary codec (libopenh264)")
            primary_cmd = [
                "ffmpeg",
                "-i", input_path,
                "-c:v", "libopenh264",
                "-b:v", "3M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-y",
                output_path
            ]
        elif use_libx264:
            logger.info("Starting FFmpeg transcoding with primary codec (libx264)")
            primary_cmd = [
                "ffmpeg",
                "-i", input_path,
                "-c:v", "libx264",
                "-b:v", "3M",
                "-maxrate", "3M",
                "-bufsize", "6M",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-y",
                output_path
            ]
        else:
            logger.info("Starting FFmpeg transcoding with fallback codec (mpeg4)")
            primary_cmd = [
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
