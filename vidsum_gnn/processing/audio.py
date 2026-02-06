import os
import asyncio
from vidsum_gnn.core.config import settings
from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)

async def extract_audio_segment(video_path: str, start: float, end: float, output_path: str) -> str:
    """
    Extract audio segment for a shot.
    """
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid audio segment duration: start={start}, end={end}")
    # Prefer WAV PCM for maximum compatibility (Whisper + torchaudio).
    # MP3 decoding can be flaky depending on backend availability.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel", "error",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vn",  # No video
        "-ac", "1",  # Mono
        "-ar", "16000",  # 16kHz for Whisper
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "-y",
        output_path,
    ]

    logger.debug(f"Audio seg cmd: {' '.join(cmd)}")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {stderr.decode(errors='ignore')[:500]}")

    try:
        sz = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        if sz < 2048:
            logger.warning(f"Extracted audio segment is tiny ({sz} bytes): {output_path}")
        else:
            logger.debug(f"Extracted audio segment: {output_path} ({duration:.2f}s, {sz} bytes)")
    except Exception:
        pass
        
    return output_path

async def extract_full_audio(video_path: str, output_path: str) -> str:
    """
    Extract full audio track.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel", "error",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "-y",
        output_path,
    ]

    logger.debug(f"Audio full cmd: {' '.join(cmd)}")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg full audio extraction failed: {stderr.decode(errors='ignore')[:500]}")

    try:
        sz = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        logger.debug(f"Extracted full audio: {output_path} ({sz} bytes)")
    except Exception:
        pass
        
    return output_path
