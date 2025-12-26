import os
import asyncio
import re
from typing import List, Tuple
from vidsum_gnn.core.config import settings

async def detect_shots(video_path: str, threshold: float = 0.4) -> List[Tuple[float, float]]:
    """
    Detect shots using ffmpeg scene detection filter.
    Returns list of (start_time, end_time) tuples.
    """
    # ffmpeg -i input.mp4 -filter:v "select='gt(scene,0.4)',showinfo" -f null -
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null",
        "-"
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    
    # Parse stderr for showinfo lines
    # [Parsed_showinfo_1 @ ...] n:   0 pts:  12345 pts_time:0.4115 ...
    output = stderr.decode()
    shot_boundaries = [0.0]
    
    for line in output.split('\n'):
        if "pts_time" in line and "showinfo" in line:
            match = re.search(r"pts_time:([\d\.]+)", line)
            if match:
                shot_boundaries.append(float(match.group(1)))
                
    # Get total duration to close the last shot
    # We might need to pass duration in or probe it again, 
    # but for now let's assume the caller handles the final boundary or we append a large number
    # Better: probe video here to get end time
    from vidsum_gnn.processing.video import probe_video
    meta = await probe_video(video_path)
    duration = meta["duration"]
    
    if shot_boundaries[-1] < duration:
        shot_boundaries.append(duration)
        
    shots = []
    for i in range(len(shot_boundaries) - 1):
        start = shot_boundaries[i]
        end = shot_boundaries[i+1]
        # Filter extremely short shots (< 0.5s)
        if end - start > 0.5:
            shots.append((start, end))
            
    return shots

async def extract_keyframe(video_path: str, timestamp: float, output_path: str) -> str:
    """
    Extract a single keyframe at the given timestamp.
    """
    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
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
        raise RuntimeError(f"ffmpeg keyframe extraction failed: {stderr.decode()}")
        
    return output_path

async def sample_frames_for_shots(video_path: str, shots: List[Tuple[float, float]], video_id: str) -> List[str]:
    """
    Extract center keyframe for each shot.
    Returns list of file paths.
    """
    output_dir = os.path.join(settings.PROCESSED_DIR, video_id, "keyframes")
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = []
    paths = []
    
    for i, (start, end) in enumerate(shots):
        center = start + (end - start) / 2
        filename = f"shot_{i:04d}_{center:.2f}.jpg"
        path = os.path.join(output_dir, filename)
        paths.append(path)
        
        # We can limit concurrency here if needed, but for now let's just create tasks
        tasks.append(extract_keyframe(video_path, center, path))
        
    # Run in batches to avoid opening too many files/processes
    batch_size = 10
    for i in range(0, len(tasks), batch_size):
        await asyncio.gather(*tasks[i:i+batch_size])
        
    return paths
