import os
import asyncio
import re
from typing import List, Tuple
from vidsum_gnn.core.config import settings
from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)

async def detect_shots(video_path: str, threshold: float = 0.4) -> List[Tuple[float, float]]:
    """
    Detect shots using ffmpeg scene detection filter.
    Returns list of (start_time, end_time) tuples.
    """
    # ffmpeg -i input.mp4 -filter:v "select='gt(scene,0.4)',showinfo" -f null -
    async def _run_scene_detect(scene_threshold: float) -> str:
        # NOTE: showinfo logs at INFO level; using -loglevel error suppresses it and yields 1-shot output.
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-loglevel", "info",
            "-i", video_path,
            "-filter:v", f"select='gt(scene,{scene_threshold})',showinfo",
            "-an",
            "-f", "null",
            "-",
        ]

        logger.debug(f"Shot detect cmd: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            err = (stderr or b"").decode(errors="ignore")
            raise RuntimeError(f"ffmpeg shot detection failed: {err[:500]}")

        # showinfo is emitted to stderr, but some builds may write to stdout.
        out = (stderr or b"") + b"\n" + (stdout or b"")
        return out.decode(errors="ignore")

    # Parse ffmpeg output for showinfo lines
    # [Parsed_showinfo_1 @ ...] n:   0 pts:  12345 pts_time:0.4115 ...
    shot_boundaries = [0.0]

    output = await _run_scene_detect(threshold)
    for line in output.split("\n"):
        if "pts_time" in line and "showinfo" in line:
            match = re.search(r"pts_time:([\d\.]+)", line)
            if match:
                shot_boundaries.append(float(match.group(1)))

    # If we found no cuts, retry with a lower threshold (helps for lectures/slides)
    # Keep retries small to avoid expensive repeated ffmpeg runs.
    if len(shot_boundaries) <= 1 and threshold > 0.10:
        retry_threshold = max(0.08, float(threshold) * 0.60)
        logger.info(f"No scene cuts detected at threshold={threshold}; retrying with threshold={retry_threshold}")
        output = await _run_scene_detect(retry_threshold)
        for line in output.split("\n"):
            if "pts_time" in line and "showinfo" in line:
                match = re.search(r"pts_time:([\d\.]+)", line)
                if match:
                    shot_boundaries.append(float(match.group(1)))

    # De-dup and sort boundaries (ffmpeg can emit duplicates/out-of-order).
    shot_boundaries = sorted(set(shot_boundaries))
                
    # Get total duration to close the last shot
    # We might need to pass duration in or probe it again, 
    # but for now let's assume the caller handles the final boundary or we append a large number
    # Better: probe video here to get end time
    from vidsum_gnn.processing.video import probe_video
    meta = await probe_video(video_path)
    duration = meta["duration"]
    
    if shot_boundaries[-1] < duration:
        shot_boundaries.append(duration)

    # Clip and re-sort after adding duration.
    shot_boundaries = [b for b in shot_boundaries if 0.0 <= b <= duration]
    shot_boundaries = sorted(set(shot_boundaries))
        
    shots = []
    for i in range(len(shot_boundaries) - 1):
        start = shot_boundaries[i]
        end = shot_boundaries[i+1]
        # Filter extremely short shots (< 0.5s)
        if end - start > 0.5:
            shots.append((start, end))
    logger.info(
        f"Detected {len(shots)} shots (threshold={threshold}, duration={duration:.2f}s, boundaries={len(shot_boundaries)})"
    )
    return shots

async def extract_keyframe(video_path: str, timestamp: float, output_path: str) -> str:
    """
    Extract a single keyframe at the given timestamp.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel", "error",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y",
        output_path
    ]

    logger.debug(f"Keyframe cmd: {' '.join(cmd)}")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg keyframe extraction failed: {stderr.decode(errors='ignore')[:500]}")
        
    return output_path

async def sample_frames_for_shots(
    video_path: str,
    shots: List[Tuple[float, float]],
    video_id: str,
    frames_per_shot: int = 3,
) -> Tuple[List[str], List[int]]:
    """
    Extract multiple representative frames per shot (evenly spaced inside the shot).
    Returns:
        flat_paths: list of extracted frame file paths
        frames_per_shot_list: how many frames were taken for each shot (matches len(shots))
    """
    output_dir = os.path.join(settings.PROCESSED_DIR, video_id, "keyframes")
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = []
    paths: List[str] = []
    counts: List[int] = []

    max_frames = max(1, frames_per_shot)
    for i, (start, end) in enumerate(shots):
        duration = max(0.0, end - start)
        # For very short shots, fall back to a single center frame
        n_frames = 1 if duration < 0.6 else max_frames
        # Evenly spaced timestamps inside the shot; avoid the exact edges
        stamps = [start + duration * (j + 1) / (n_frames + 1) for j in range(n_frames)]
        counts.append(len(stamps))

        for j, ts in enumerate(stamps):
            filename = f"shot_{i:04d}_{j:02d}_{ts:.2f}.jpg"
            path = os.path.join(output_dir, filename)
            paths.append(path)
            tasks.append(extract_keyframe(video_path, ts, path))
        
    logger.info(f"Extracting {len(tasks)} keyframes across {len(shots)} shots (batch_size=10, frames_per_shot≈{max_frames})")

    # Run in batches to avoid opening too many files/processes
    batch_size = 10
    for i in range(0, len(tasks), batch_size):
        await asyncio.gather(*tasks[i:i+batch_size])
        
    return paths, counts
