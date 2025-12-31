"""
YouTube Playlist -> Video Summarization Dataset
Pipeline for downloading, shot detection, feature extraction, scoring, and storage.
"""

import gc
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import librosa
import numpy as np
from scenedetect import ContentDetector, detect
from scipy.ndimage import gaussian_filter1d


# ============================================================================
# Storage Layout
# ============================================================================

RAW_YOUTUBE_DIR = Path("model/data/raw/youtube")
RAW_UGC_DIR = Path("model/data/raw/ugc")
PROCESSED_OUTPUT_DIR = Path("model/data/processed/features")


# ============================================================================
# Data Schema
# ============================================================================


@dataclass
class ShotFeatures:
    motion: float
    speech: float
    scene_change: float
    audio_energy: float
    object_count: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Shot:
    start: float
    end: float
    features: ShotFeatures
    importance: float
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "features": self.features.to_dict(),
            "importance": self.importance,
            "rank": self.rank,
        }


@dataclass
class VideoDataset:
    video_id: str
    duration: float
    domain: str
    shots: List[Shot]

    def to_dict(self) -> Dict[str, object]:
        return {
            "video_id": self.video_id,
            "duration": self.duration,
            "domain": self.domain,
            "shots": [s.to_dict() for s in self.shots],
        }


# ============================================================================
# Domain Weights & Performance Config
# ============================================================================

DOMAIN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "lecture": {"motion": 0.1, "speech": 0.5, "scene_change": 0.1, "audio_energy": 0.2, "object_count": 0.1},
    "interview": {"motion": 0.2, "speech": 0.4, "scene_change": 0.1, "audio_energy": 0.2, "object_count": 0.1},
    "sports": {"motion": 0.5, "speech": 0.1, "scene_change": 0.2, "audio_energy": 0.1, "object_count": 0.1},
    "documentary": {"motion": 0.25, "speech": 0.25, "scene_change": 0.2, "audio_energy": 0.2, "object_count": 0.1},
    "gaming": {"motion": 0.6, "speech": 0.15, "scene_change": 0.15, "audio_energy": 0.05, "object_count": 0.05},
    "vlog": {"motion": 0.2, "speech": 0.5, "scene_change": 0.1, "audio_energy": 0.15, "object_count": 0.05},
    "default": {"motion": 0.35, "speech": 0.25, "scene_change": 0.2, "audio_energy": 0.1, "object_count": 0.1},
}

MOTION_DOWNSCALE_WIDTH = 640  # None to disable
MOTION_DISABLED_DOMAINS = {"lecture", "interview"}
SHOT_GC_INTERVAL = 10


# ============================================================================
# Helpers
# ============================================================================


def clear_memory() -> None:
    try:
        gc.collect()
    except Exception:
        pass


def chunked(iterable: Iterable, size: int) -> Iterable[List]:
    buf: List = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def download_playlist(playlist_url: str, playlist_name: str, output_base: Path = RAW_YOUTUBE_DIR, max_items: int = 10) -> List[Path]:
    """Download videos into raw/youtube/<playlist_name>."""

    output_dir = output_base / playlist_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "-f",
        "best[ext=mp4]",
        "-o",
        f"{output_dir}/%(id)s.%(ext)s",
        "--write-info-json",
    ]
    if max_items is not None:
        cmd += ["--playlist-items", f"1-{max_items}"]

    cmd.append(playlist_url)
    subprocess.run(cmd, check=True)
    return list(output_dir.glob("*.mp4"))


# ============================================================================
# Step 1: Audio Extraction
# ============================================================================


def extract_audio(video_path: Path, output_dir: str = "model/data/raw/youtube/audio") -> Path:
    """Extract mono 16k WAV from video."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    audio_path = Path(output_dir) / f"{video_path.stem}.wav"

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
        "-y",
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


# ============================================================================
# Step 2: Shot Detection
# ============================================================================


def detect_shots(video_path: Path, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """Detect shot boundaries using scenedetect; fallback to single-shot."""

    try:
        scenes = detect(str(video_path), ContentDetector(threshold=threshold))
        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
    except Exception as exc:
        print(f"Warning: Shot detection failed - {exc}")
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        return [(0.0, duration)]


# ============================================================================
# Step 3: Feature Extraction
# ============================================================================


def compute_motion_score(video_path: Path, start: float, end: float) -> float:
    """Compute optical flow magnitude for the shot (every frame, optional downscale)."""

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return 0.0

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if MOTION_DOWNSCALE_WIDTH and prev_gray.shape[1] > MOTION_DOWNSCALE_WIDTH:
        new_h = int(prev_gray.shape[0] * (MOTION_DOWNSCALE_WIDTH / prev_gray.shape[1]))
        prev_gray = cv2.resize(prev_gray, (MOTION_DOWNSCALE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    scores: List[float] = []
    frame_count = 0
    max_frames = int((end - start) * fps) if fps > 0 else 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if MOTION_DOWNSCALE_WIDTH and gray.shape[1] > MOTION_DOWNSCALE_WIDTH:
            new_h = int(gray.shape[0] * (MOTION_DOWNSCALE_WIDTH / gray.shape[1]))
            gray = cv2.resize(gray, (MOTION_DOWNSCALE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        mag = float(np.mean(np.linalg.norm(flow, axis=2)))
        scores.append(mag)
        prev_gray = gray
        frame_count += 1

    cap.release()
    return float(np.mean(scores)) if scores else 0.0


def compute_speech_activity(audio_path: Path, start: float, end: float) -> float:
    """Compute speech activity via RMS energy."""

    duration = end - start
    if duration <= 0:
        return 0.0

    y, _ = librosa.load(str(audio_path), sr=16000, offset=start, duration=duration)
    rms = librosa.feature.rms(y=y)[0]
    return float(np.mean(rms))


def compute_audio_energy(audio_path: Path, start: float, end: float) -> float:
    """Compute mean absolute audio energy."""

    duration = end - start
    if duration <= 0:
        return 0.0

    y, _ = librosa.load(str(audio_path), sr=16000, offset=start, duration=duration)
    return float(np.mean(np.abs(y)))


def extract_shot_features(video_path: Path, audio_path: Path, start: float, end: float, domain: str) -> ShotFeatures:
    """Extract shot-level features; skip motion for configured domains."""

    motion_val = 0.0 if domain in MOTION_DISABLED_DOMAINS else compute_motion_score(video_path, start, end)
    return ShotFeatures(
        motion=motion_val,
        speech=compute_speech_activity(audio_path, start, end),
        scene_change=1.0,
        audio_energy=compute_audio_energy(audio_path, start, end),
        object_count=1.0,
    )


# ============================================================================
# Step 4: Importance Scoring with Temporal Smoothing
# ============================================================================


def normalize_features(shots_features: List) -> List[Dict[str, float]]:
    if not shots_features:
        return []

    if isinstance(shots_features[0], ShotFeatures):
        features_list = [
            {
                "motion": s.motion,
                "speech": s.speech,
                "scene_change": s.scene_change,
                "audio_energy": s.audio_energy,
                "object_count": s.object_count,
            }
            for s in shots_features
        ]
    else:
        features_list = shots_features

    features_dict = {
        "motion": [s["motion"] for s in features_list],
        "speech": [s["speech"] for s in features_list],
        "scene_change": [s["scene_change"] for s in features_list],
        "audio_energy": [s["audio_energy"] for s in features_list],
        "object_count": [s["object_count"] for s in features_list],
    }

    normalized: Dict[str, np.ndarray] = {}
    for key, values in features_dict.items():
        arr = np.array(values, dtype=float)
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val > 1e-6:
            normalized[key] = (arr - min_val) / (max_val - min_val)
        else:
            normalized[key] = np.zeros_like(arr)

    result: List[Dict[str, float]] = []
    for i in range(len(features_list)):
        result.append({
            "motion": float(normalized["motion"][i]),
            "speech": float(normalized["speech"][i]),
            "scene_change": float(normalized["scene_change"][i]),
            "audio_energy": float(normalized["audio_energy"][i]),
            "object_count": float(normalized["object_count"][i]),
        })

    return result


def compute_importance(features, weights: Optional[Dict[str, float]] = None, domain: Optional[str] = None) -> float:
    if domain is not None:
        weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["default"])
    elif weights is None:
        weights = {"motion": 0.2, "speech": 0.2, "scene_change": 0.2, "audio_energy": 0.2, "object_count": 0.2}

    if isinstance(features, dict):
        motion = features["motion"]
        speech = features["speech"]
        scene_change = features["scene_change"]
        audio_energy = features["audio_energy"]
        object_count = features["object_count"]
    else:
        motion = features.motion
        speech = features.speech
        scene_change = features.scene_change
        audio_energy = features.audio_energy
        object_count = features.object_count

    score = (
        weights["motion"] * motion
        + weights["speech"] * speech
        + weights["scene_change"] * scene_change
        + weights["audio_energy"] * audio_energy
        + weights["object_count"] * object_count
    )
    return float(score)


def smooth_importance_scores(scores: List[float], sigma: float = 2.0) -> List[float]:
    if len(scores) < 3:
        return scores
    smoothed = gaussian_filter1d(scores, sigma=sigma)
    return smoothed.tolist()


def assign_ranks(shots: List[Shot]) -> List[Shot]:
    if not shots:
        return []

    ranked = [shot for shot in shots]
    sorted_indices = sorted(range(len(ranked)), key=lambda i: ranked[i].importance, reverse=True)

    for rank, idx in enumerate(sorted_indices, 1):
        old_shot = ranked[idx]
        ranked[idx] = Shot(
            start=old_shot.start,
            end=old_shot.end,
            features=old_shot.features,
            importance=old_shot.importance,
            rank=rank,
        )

    return ranked


# ============================================================================
# Step 5: Full Pipeline
# ============================================================================


def process_video(video_path: Path, audio_path: Path, domain: str = "default", smooth_sigma: float = 2.0) -> VideoDataset:
    weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["default"])

    shot_boundaries = detect_shots(video_path)
    print(f"Detected {len(shot_boundaries)} shots")

    shots_features: List[ShotFeatures] = []
    for i, (start, end) in enumerate(shot_boundaries):
        print(f"Processing shot {i + 1}/{len(shot_boundaries)}..." + (" (skip motion)" if domain in MOTION_DISABLED_DOMAINS else ""))
        features = extract_shot_features(video_path, audio_path, start, end, domain)
        shots_features.append(features)
        if (i + 1) % SHOT_GC_INTERVAL == 0:
            clear_memory()

    shots_features = normalize_features(shots_features)

    importance_scores = [compute_importance(f, weights) for f in shots_features]
    importance_scores = smooth_importance_scores(importance_scores, sigma=smooth_sigma)

    shots = [
        Shot(start=start, end=end, features=ShotFeatures(**feat), importance=imp)
        for (start, end), feat, imp in zip(shot_boundaries, shots_features, importance_scores)
    ]

    shots = assign_ranks(shots)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()
    clear_memory()

    return VideoDataset(video_id=video_path.stem, duration=duration, domain=domain, shots=shots)


def build_dataset(
    video_dir: Path = RAW_YOUTUBE_DIR,
    output_path: Optional[Path] = None,
    max_per_playlist: Optional[int] = 10,
    batch_size: int = 3,
    domain_map: Optional[Dict[str, Dict[str, str]]] = None,
    playlist_filter: Optional[str] = None,
    save_frequency: str = "playlist",
    include_ugc: bool = False,
) -> List[VideoDataset]:
    video_root = Path(video_dir)
    dataset: List[VideoDataset] = []

    if output_path is None:
        output_path = PROCESSED_OUTPUT_DIR / "complete_dataset.json"

    if not video_root.exists():
        print(f"⚠️  Video root not found: {video_root}")
        return dataset

    playlist_dirs = [p for p in video_root.iterdir() if p.is_dir()]
    if playlist_filter:
        playlist_dirs = [p for p in playlist_dirs if p.name == playlist_filter]
    
    # Process YouTube playlists
    for playlist_dir in sorted(playlist_dirs):
        playlist_name = playlist_dir.name
        domain = domain_map.get(playlist_name, {}).get("domain", "default") if domain_map else "default"
        video_paths = sorted(playlist_dir.glob("*.mp4"))
        if max_per_playlist is not None:
            video_paths = video_paths[:max_per_playlist]

        if not video_paths:
            print(f"⚠️  No videos found in {playlist_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Playlist: {playlist_name} ({domain})")
        print(f"Found {len(video_paths)} videos (processing up to {len(video_paths)})")

        for batch in chunked(video_paths, batch_size):
            for video_path in batch:
                print(f"\nProcessing: {video_path.name}")
                try:
                    audio_path = extract_audio(video_path)
                    video_data = process_video(video_path, audio_path, domain=domain)
                    dataset.append(video_data)
                    print(f"✓ Completed: {len(video_data.shots)} shots")
                    if save_frequency == "video":
                        save_dataset_structure(dataset, PROCESSED_OUTPUT_DIR)
                except Exception as exc:
                    print(f"✗ Error processing {video_path.name}: {exc}")
                finally:
                    clear_memory()
            clear_memory()

        if save_frequency in {"playlist", "video"} and dataset:
            save_dataset_structure(dataset, PROCESSED_OUTPUT_DIR)

    # Process UGC dataset if requested
    if include_ugc and RAW_UGC_DIR.exists():
        ugc_videos = sorted(RAW_UGC_DIR.glob("*.mp4"))
        if ugc_videos:
            print(f"\n{'='*60}")
            print(f"Dataset: UGC (User-Generated Content)")
            print(f"Found {len(ugc_videos)} videos (processing all)")
            
            for batch in chunked(ugc_videos, batch_size):
                for video_path in batch:
                    print(f"\nProcessing: {video_path.name}")
                    domain = get_ugc_domain_from_filename(video_path.name)
                    try:
                        audio_path = extract_audio(video_path)
                        video_data = process_video(video_path, audio_path, domain=domain)
                        dataset.append(video_data)
                        print(f"✓ Completed: {len(video_data.shots)} shots (domain: {domain})")
                        if save_frequency == "video":
                            save_dataset_structure(dataset, PROCESSED_OUTPUT_DIR)
                    except Exception as exc:
                        print(f"✗ Error processing {video_path.name}: {exc}")
                    finally:
                        clear_memory()
                clear_memory()
            
            if save_frequency in {"playlist", "video"} and dataset:
                save_dataset_structure(dataset, PROCESSED_OUTPUT_DIR)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([v.to_dict() for v in dataset], f, indent=2)
        print(f"\n✓ Dataset saved to {output_path}")
        print(f"Total videos: {len(dataset)}")
        print(f"Total shots: {sum(len(v.shots) for v in dataset)}")

    return dataset


def get_ugc_domain_from_filename(filename: str) -> str:
    """Extract domain from UGC filename pattern (Gaming_*, Sports_*, Vlog_*)."""
    if filename.startswith("Gaming_"):
        return "gaming"
    elif filename.startswith("Sports_"):
        return "sports"
    elif filename.startswith("Vlog_"):
        return "vlog"
    else:
        return "default"


# ============================================================================
# Top 5 Major YouTube Playlists Configuration
# ============================================================================


TOP_5_PLAYLISTS: Dict[str, Dict[str, str]] = {
    "TED-Talks": {
        "url": "https://www.youtube.com/playlist?list=PLQltO7RlbjPJnbfHLsFJWP-DYnWPugUZ7",
        "domain": "lecture",
        "description": "TED Talks curated playlist",
    },
    "Kurzgesagt": {
        "url": "https://www.youtube.com/playlist?list=PLfcYWTX53e-EKS_0V-8qaJHGscpdQ_eK1",
        "domain": "documentary",
        "description": "Educational science videos",
    },
    # "BBC-Breaking-News": {
    #     "url": "https://www.youtube.com/playlist?list=PLS3XGZxi7cBVTzEE4Sim9UuNKnUJq9Vkh",
    #     "domain": "documentary",
    #     "description": "News and current events",
    # },
    "ESPN-Highlights": {
        "url": "https://www.youtube.com/playlist?list=PL87LlAF-2PIwKpIUaKO4_p5QNmjxhYUFG",
        "domain": "sports",
        "description": "Sports highlights",
    },
    # "BBC-Learning": {
    #     "url": "https://www.youtube.com/playlist?list=PLcetZ6gSk969oGvAI0e4_PgVnlGbm64bp",
    #     "domain": "documentary",
    #     "description": "BBC educational content",
    # },
}


# ============================================================================
# Dataset Management & Testing
# ============================================================================


def validate_dataset(dataset: List[VideoDataset]) -> Dict[str, object]:
    if not dataset:
        return {"valid": False, "error": "Empty dataset"}

    stats = {
        "valid": True,
        "num_videos": len(dataset),
        "total_shots": sum(len(v.shots) for v in dataset),
        "total_duration": sum(v.duration for v in dataset),
        "avg_shots_per_video": sum(len(v.shots) for v in dataset) / len(dataset),
        "avg_duration_per_video": sum(v.duration for v in dataset) / len(dataset),
        "domains": list(set(v.domain for v in dataset)),
        "importance_stats": {
            "min": min(s.importance for v in dataset for s in v.shots),
            "max": max(s.importance for v in dataset for s in v.shots),
            "mean": float(np.mean([s.importance for v in dataset for s in v.shots])),
            "std": float(np.std([s.importance for v in dataset for s in v.shots])),
        },
    }
    return stats


def _assign_split(idx: int) -> str:
    """Deterministically assign a split (train/val/test) based on index for incremental saves."""
    mod = idx % 10
    if mod < 6:
        return "train"
    if mod < 8:
        return "val"
    return "test"


def save_dataset_structure(dataset: List[VideoDataset], output_dir: Path = PROCESSED_OUTPUT_DIR) -> Dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_dir = output_path / "metadata"
    features_dir = output_path / "features"
    splits_dir = output_path / "splits"
    for d in [metadata_dir, features_dir, splits_dir]:
        d.mkdir(exist_ok=True)

    dataset_file = output_path / "complete_dataset.json"
    with open(dataset_file, "w") as f:
        json.dump([v.to_dict() for v in dataset], f, indent=2)

    metadata_videos = []
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for idx, v in enumerate(dataset):
        split = _assign_split(idx)
        splits[split].append(v.video_id)
        metadata_videos.append(
            {
                "video_id": v.video_id,
                "duration": v.duration,
                "domain": v.domain,
                "split": split,
                "num_shots": len(v.shots),
                "importance_stats": {
                    "min": float(min(s.importance for s in v.shots)) if v.shots else 0.0,
                    "max": float(max(s.importance for s in v.shots)) if v.shots else 0.0,
                    "mean": float(np.mean([s.importance for s in v.shots])) if v.shots else 0.0,
                },
            }
        )

    metadata = {
        "num_videos": len(dataset),
        "videos": metadata_videos,
    }
    metadata_file = metadata_dir / "dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    for video in dataset:
        video_features = {
            "video_id": video.video_id,
            "duration": video.duration,
            "shots": [],
        }
        for shot in video.shots:
            video_features["shots"].append(
                {
                    "start": shot.start,
                    "end": shot.end,
                    "importance": shot.importance,
                    "rank": shot.rank,
                    "features": shot.features.to_dict(),
                }
            )

        features_file = features_dir / f"{video.video_id}_features.json"
        with open(features_file, "w") as f:
            json.dump(video_features, f, indent=2)

    splits_file = splits_dir / "train_val_test_split.json"
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)

    return {
        "dataset": dataset_file,
        "metadata": metadata_file,
        "features_dir": features_dir,
        "splits": splits_file,
    }


def test_dataset(dataset: List[VideoDataset]) -> bool:
    print("\n" + "=" * 70)
    print("🧪 DATASET TESTING")
    print("=" * 70)

    tests_passed = 0
    tests_total = 0

    tests_total += 1
    if len(dataset) > 0:
        print("✓ Test 1: Dataset is non-empty")
        tests_passed += 1
    else:
        print("✗ Test 1: Dataset is empty")

    tests_total += 1
    all_valid = all(hasattr(v, "video_id") and hasattr(v, "duration") and hasattr(v, "domain") and hasattr(v, "shots") for v in dataset)
    if all_valid:
        print("✓ Test 2: All videos have required fields")
        tests_passed += 1
    else:
        print("✗ Test 2: Some videos missing required fields")

    tests_total += 1
    all_shots_valid = all(hasattr(s, "start") and hasattr(s, "end") and hasattr(s, "importance") and 0 <= s.importance <= 1 for v in dataset for s in v.shots)
    if all_shots_valid:
        print("✓ Test 3: All shots have valid structure and importance in [0,1]")
        tests_passed += 1
    else:
        print("✗ Test 3: Some shots have invalid structure")

    tests_total += 1
    all_temporal_valid = all(s.start < s.end for v in dataset for s in v.shots)
    if all_temporal_valid:
        print("✓ Test 4: All shots have valid temporal boundaries")
        tests_passed += 1
    else:
        print("✗ Test 4: Some shots have invalid temporal boundaries")

    tests_total += 1
    valid_domains = {"lecture", "interview", "sports", "documentary", "default"}
    all_domains_valid = all(v.domain in valid_domains for v in dataset)
    if all_domains_valid:
        print("✓ Test 5: All videos have valid domain labels")
        tests_passed += 1
    else:
        print("✗ Test 5: Some videos have invalid domain labels")

    tests_total += 1
    stats = validate_dataset(dataset)
    print("✓ Test 6: Dataset statistics computed")
    print(f"   - Videos: {stats['num_videos']}")
    print(f"   - Total shots: {stats['total_shots']}")
    print(f"   - Avg shots/video: {stats['avg_shots_per_video']:.1f}")
    print(f"   - Duration: {stats['total_duration']:.0f}s ({stats['total_duration']/3600:.1f}h)")
    tests_passed += 1

    print("\n" + "=" * 70)
    print(f"TESTS PASSED: {tests_passed}/{tests_total}")
    print("=" * 70 + "\n")

    return tests_passed == tests_total


# ============================================================================
# Main Execution
# ============================================================================


if __name__ == "__main__":
    import sys

    print("\n" + "=" * 70)
    print("📺 VIDEO SUMMARIZATION DATASET BUILDER (UGC Only)")
    print("=" * 70)

    # UGC-only mode
    BASE_VIDEO_DIR = RAW_UGC_DIR
    BASE_OUTPUT_DIR = PROCESSED_OUTPUT_DIR
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAX_VIDEOS_PER_PLAYLIST = None  # Process all UGC videos

    target_playlist = sys.argv[1] if len(sys.argv) > 1 else None
    save_frequency = sys.argv[2] if len(sys.argv) > 2 else "playlist"
    
    if save_frequency not in {"playlist", "video"}:
        save_frequency = "playlist"

    print("✓ Processing UGC dataset only (Gaming, Sports, Vlog)\n")

    all_datasets = build_dataset(
        video_dir=BASE_VIDEO_DIR,
        output_path=BASE_OUTPUT_DIR / "complete_dataset.json",
        max_per_playlist=MAX_VIDEOS_PER_PLAYLIST,
        batch_size=3,
        domain_map=TOP_5_PLAYLISTS,
        playlist_filter=target_playlist,
        save_frequency=save_frequency,
        include_ugc=False,  # UGC is primary source now
    )

    if all_datasets:
        print("\n" + "=" * 70)
        print("DATASET VALIDATION & STORAGE")
        print("=" * 70)

        is_valid = test_dataset(all_datasets)

        if is_valid:
            print("\n💾 Saving dataset structure...")
            saved_paths = save_dataset_structure(all_datasets, PROCESSED_OUTPUT_DIR)

            print("\n✓ Dataset saved successfully!")
            for name, path in saved_paths.items():
                print(f"   {name}: {path}")

            stats = validate_dataset(all_datasets)
            print("\n" + "=" * 70)
            print("📊 FINAL DATASET STATISTICS")
            print("=" * 70)
            print(f"Videos: {stats['num_videos']}")
            print(f"Total shots: {stats['total_shots']}")
            print(f"Total duration: {stats['total_duration']/3600:.2f} hours")
            print(f"Avg shots/video: {stats['avg_shots_per_video']:.1f}")
            print(f"Avg duration/video: {stats['avg_duration_per_video']:.0f}s")
            print(f"Domains: {', '.join(stats['domains'])}")
            print(f"Importance range: [{stats['importance_stats']['min']:.3f}, {stats['importance_stats']['max']:.3f}]")
            print(f"Mean importance: {stats['importance_stats']['mean']:.3f} ± {stats['importance_stats']['std']:.3f}")
            print("=" * 70 + "\n")
        else:
            print("\n✗ Dataset validation failed!")
            sys.exit(1)
    else:
        print("\n⚠️  No datasets were successfully created!")
        print("\nUsage:")
        print("  python youtube_dataset.py                 # Process all UGC videos")
        print("  python youtube_dataset.py <domain>        # Process specific domain (Gaming, Sports, Vlog)")
        print("\nAvailable UGC domains:")
        print("  - Gaming   (35 videos, high motion)")
        print("  - Sports   (55 videos, dynamic scenes)")
        print("  - Vlog     (35 videos, speech-heavy)")
