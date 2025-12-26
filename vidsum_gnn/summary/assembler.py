import os
import asyncio
from typing import List, Dict
from vidsum_gnn.core.config import settings

async def assemble_summary(video_path: str, selected_shots: List[Dict], output_path: str) -> str:
    """
    Assemble selected shots into a summary video.
    Uses concat demuxer approach: extract clips -> concat.
    """
    # 1. Create temp directory for clips
    video_id = selected_shots[0]['video_id'] if selected_shots else "temp"
    clips_dir = os.path.join(settings.TEMP_DIR, video_id, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    
    clip_paths = []
    
    # 2. Extract each clip
    # We can run this in parallel
    tasks = []
    
    for i, shot in enumerate(selected_shots):
        start = shot['start_sec']
        duration = shot['duration_sec']
        clip_path = os.path.join(clips_dir, f"clip_{i:04d}.mp4")
        clip_paths.append(clip_path)
        
        # ffmpeg -ss start -i input -t duration -c copy ...
        # Re-encoding is safer for concat to avoid timestamp issues, but slower.
        # We'll re-encode to a standard intermediate format to ensure smooth concat.
        cmd = [
            "ffmpeg",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "ultrafast", # Speed over size for intermediate
            "-c:a", "aac",
            "-y",
            clip_path
        ]
        
        tasks.append(asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        ))
        
    # Run extraction
    # Limit concurrency
    batch_size = 5
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        procs = await asyncio.gather(*batch)
        for p in procs:
            await p.communicate()
            
    # 3. Create concat list file
    list_path = os.path.join(clips_dir, "concat_list.txt")
    with open(list_path, "w") as f:
        for p in clip_paths:
            # Escape path for ffmpeg
            p_escaped = p.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{p_escaped}'\n")
            
    # 4. Concat
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy", # Copy since we re-encoded clips
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
        raise RuntimeError(f"ffmpeg concat failed: {stderr.decode()}")
        
    # Cleanup clips? Maybe later or rely on temp dir cleanup
    
    return output_path
