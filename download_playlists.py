#!/usr/bin/env python
"""
YouTube Playlist Downloader
Downloads videos from the top 5 playlists with progress tracking
"""

import subprocess
import sys
from pathlib import Path
from youtube_dataset import TOP_5_PLAYLISTS

def download_playlist(playlist_name: str, playlist_url: str, output_dir: str, limit: int = 3):
    """Download videos from a YouTube playlist using yt-dlp"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # yt-dlp command with best MP4 format, limited to N videos
    output_template = str(output_path / "%(id)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",  # Best MP4 format
        "-o", output_template,
        "--no-warnings",
        "-q",  # Quiet mode
    ]
    
    # Add playlist items limit
    if limit > 0:
        cmd.extend(["--playlist-items", f"1-{limit}"])
    
    cmd.append(playlist_url)
    
    print(f"\n{'='*70}")
    print(f"Downloading: {playlist_name}")
    print(f"{'='*70}")
    print(f"URL: {playlist_url}")
    print(f"Output: {output_path}")
    print(f"Limit: {limit} videos")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Count downloaded files
            video_count = len(list(output_path.glob("*.mp4")))
            print(f"✓ Successfully downloaded {video_count} videos")
            return True
        else:
            print(f"✗ Download failed with error:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("✗ yt-dlp not found. Please install with: pip install yt-dlp")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Download videos from all top 5 playlists"""
    print("\n" + "="*70)
    print("YOUTUBE PLAYLIST DOWNLOADER")
    print("="*70)
    
    base_video_dir = Path("model/data/videos")
    base_video_dir.mkdir(parents=True, exist_ok=True)
    
    # Ask user for number of videos to download per playlist
    try:
        limit = int(input("\nHow many videos per playlist? (default: 3, max: 20): ") or "3")
        limit = min(max(1, limit), 20)  # Clamp between 1 and 20
    except ValueError:
        limit = 3
    
    print(f"\nDownloading {limit} videos per playlist...")
    
    success_count = 0
    for playlist_name, config in TOP_5_PLAYLISTS.items():
        output_dir = base_video_dir / playlist_name
        
        success = download_playlist(
            playlist_name,
            config['url'],
            str(output_dir),
            limit=limit
        )
        
        if success:
            success_count += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully downloaded from {success_count}/{len(TOP_5_PLAYLISTS)} playlists")
    
    # Count total videos
    total_videos = 0
    for playlist_dir in base_video_dir.iterdir():
        if playlist_dir.is_dir():
            video_count = len(list(playlist_dir.glob("*.mp4")))
            total_videos += video_count
            print(f"  {playlist_dir.name:20} : {video_count:3} videos")
    
    print(f"{'='*70}")
    print(f"Total videos: {total_videos}")
    print(f"\nNext step: Run 'python youtube_dataset.py' to process videos")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
