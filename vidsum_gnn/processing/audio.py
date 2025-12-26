import os
import asyncio
from vidsum_gnn.core.config import settings

async def extract_audio_segment(video_path: str, start: float, end: float, output_path: str) -> str:
    """
    Extract audio segment for a shot.
    """
    duration = end - start
    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vn", # No video
        "-acodec", "pcm_s16le",
        "-ar", "16000", # 16kHz for Wav2Vec2
        "-ac", "1", # Mono
        "-y",
        output_path
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {stderr.decode()}")
        
    return output_path

async def extract_full_audio(video_path: str, output_path: str) -> str:
    """
    Extract full audio track.
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg full audio extraction failed: {stderr.decode()}")
        
    return output_path
