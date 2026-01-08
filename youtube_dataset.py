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
ENABLE_ASR = True  # Toggle to True to run Whisper ASR; keep False to skip


import cv2
import librosa
import numpy as np
from scenedetect import ContentDetector, detect
from scipy.ndimage import gaussian_filter1d


# ============================================================================
# Storage Layout
# ============================================================================

RAW_YOUTUBE_DIR = Path("model/data/raw/youtube")
RAW_TVSUMME_DIR = Path("model/data/raw/tvsum")
RAW_SUMME_DIR = Path("model/data/raw/summe")

UNIFIED_OUTPUT_DIR = Path("model/data/processed/unified_dataset")


# ============================================================================
# Data Schema
# ============================================================================


@dataclass
class ShotFeatures:
    # Temporal
    duration: float
    relative_position: float  # shot_index / total_shots
    
    # Motion (descriptive, not scored)
    motion_mean: float
    motion_std: float
    motion_peak: float
    
    # Audio (rich descriptors)
    rms_energy: float
    rms_delta: float
    spectral_flux: float
    pitch_mean: float
    pitch_std: float
    silence_ratio: float
    
    # Visual change
    scene_cut_strength: float
    color_hist_delta: float
    
    # Binary silence flag (explicit, helps model ignore noise-only shots)
    is_silent: bool = False

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Shot:
    start: float
    end: float
    features: ShotFeatures
    label: Optional[float] = None  # For supervised learning (0/1 or score)
    rank: Optional[int] = None     # For reference only
    transcription: Optional[str] = None
    asr_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "features": self.features.to_dict(),
            "label": self.label,
            "rank": self.rank,
            "transcription": self.transcription,
            "asr_confidence": self.asr_confidence,
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
    "music": {"motion": 0.4, "speech": 0.3, "scene_change": 0.15, "audio_energy": 0.1, "object_count": 0.05},
    "movie_trailer": {"motion": 0.45, "speech": 0.2, "scene_change": 0.2, "audio_energy": 0.1, "object_count": 0.05},
    "movie_clip": {"motion": 0.5, "speech": 0.25, "scene_change": 0.15, "audio_energy": 0.05, "object_count": 0.05},
    "vlog": {"motion": 0.2, "speech": 0.5, "scene_change": 0.1, "audio_energy": 0.15, "object_count": 0.05},
    "default": {"motion": 0.35, "speech": 0.25, "scene_change": 0.2, "audio_energy": 0.1, "object_count": 0.1},
    # TVSum / SumMe domains are treated like default for weighting
    "tvsum": {"motion": 0.35, "speech": 0.25, "scene_change": 0.2, "audio_energy": 0.1, "object_count": 0.1},
    "summe": {"motion": 0.35, "speech": 0.25, "scene_change": 0.2, "audio_energy": 0.1, "object_count": 0.1},
}

MOTION_DOWNSCALE_WIDTH = 640  # None to disable
MOTION_FRAME_SKIP = 1         # Process every Nth frame for motion to save RAM
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


# Optional Whisper ASR helpers (kept dormant unless ENABLE_ASR=True)
_whisper_model_cache = None


def _get_asr_model(model_name: str = "small"):
    """Lazy-load Whisper model; returns None on import failure."""
    global _whisper_model_cache
    if _whisper_model_cache is not None:
        return _whisper_model_cache
    try:
        import whisper  # type: ignore
        _whisper_model_cache = whisper.load_model(model_name)
    except Exception as exc:
        print(f"⚠️  Whisper ASR unavailable: {exc}")
        _whisper_model_cache = None
    return _whisper_model_cache


def transcribe_segment(audio_path: Path, start: float, end: float, model_name: str = "small") -> Tuple[str, float]:
    """Transcribe [start, end] of an audio file. Returns (text, confidence ~ exp(logprob))."""
    duration = max(0.0, end - start)
    if duration <= 0:
        return "", 0.0

    model = _get_asr_model(model_name)
    if model is None:
        return "", 0.0

    try:
        import whisper  # type: ignore

        sample_rate = whisper.audio.SAMPLE_RATE
        audio = whisper.load_audio(str(audio_path))
        start_idx = int(start * sample_rate)
        end_idx = int(end * sample_rate)
        segment = audio[start_idx:end_idx]
        if segment.size == 0:
            return "", 0.0

        segment = whisper.pad_or_trim(segment)
        mel = whisper.log_mel_spectrogram(segment).to(model.device)
        options = whisper.DecodingOptions(
            fp16=False,
            temperature=0.0,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        result = whisper.decode(model, mel, options)
        text = result.text.strip() if result and hasattr(result, "text") else ""
        conf = float(np.exp(result.avg_logprob)) if result and hasattr(result, "avg_logprob") else 0.0
        return text, conf
    except Exception as exc:
        print(f"⚠️  ASR failed: {exc}")
        return "", 0.0


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


def download_individual_video(video_url: str, video_id: str, output_base: Path = RAW_YOUTUBE_DIR) -> Optional[Path]:
    """Download a single video by URL. Returns path to video or None if failed."""
    output_dir = output_base / "high_importance"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "--extractor-args", "youtube:player-client=android",
        "--remote-components", "ejs:github",
        "-f", "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "-o", f"{output_dir}/{video_id}.%(ext)s",
        "--write-info-json",
        video_url,
    ]

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            video_files = list(output_dir.glob(f"{video_id}.mp4"))
            return video_files[0] if video_files else None
        else:
            print(f"Warning: Failed to download {video_id}: {result.stderr}")
            return None
    except Exception as exc:
        print(f"Error downloading {video_id}: {exc}")
        return None


# ============================================================================
# Step 1: Audio Extraction
# ============================================================================


def extract_audio(video_path: Path, output_dir: str = "model/data/raw/youtube/audio") -> Path:
    """Extract mono 16k WAV from video. Returns None if no audio stream exists."""
    import shutil

    # Check if FFmpeg is installed
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg:\n"
            "  Windows: https://ffmpeg.org/download.html or 'choco install ffmpeg'\n"
            "  Add FFmpeg to your system PATH"
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    audio_path = Path(output_dir) / f"{video_path.stem}.wav"
    
    # Skip if audio already extracted
    if audio_path.exists() and audio_path.stat().st_size > 0:
        return audio_path

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

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
            return audio_path

        stderr_msg = (result.stderr or "").lower()
        # Check if no audio stream found
        if "no audio stream" in stderr_msg or "no audio streams" in stderr_msg or "does not contain any stream" in stderr_msg:
            return None  # Signal that this video has no audio

        error_msg = result.stderr or f"FFmpeg failed with code {result.returncode}"
        raise RuntimeError(f"Audio extraction failed: {error_msg}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to PATH.")


# ============================================================================
# Step 2: Shot Detection
# ============================================================================


def _get_video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    fps_safe = fps if fps > 0 else 1.0
    return float(frame_count / fps_safe) if frame_count > 0 else 0.0


def detect_shots(video_path: Path, threshold: float = 27.0, min_duration: float = 0.75) -> List[Tuple[float, float]]:
    """Detect shot boundaries using scenedetect with retries; fallback to single full-shot."""

    shots: List[Tuple[float, float]] = []
    thresholds = [threshold, 18.0, 12.0]

    try:
        for thr in thresholds:
            scenes = detect(str(video_path), ContentDetector(threshold=thr))
            shots = [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
            if shots:
                break
    except Exception as exc:
        print(f"Warning: Shot detection failed - {exc}")
        shots = []

    if not shots:
        duration = _get_video_duration(video_path)
        shots = [(0.0, duration)]

    # Merge shots that are too short (< min_duration)
    merged_shots = []
    for start, end in shots:
        duration = end - start
        if duration >= min_duration:
            merged_shots.append((start, end))
        elif merged_shots:
            prev_start, prev_end = merged_shots[-1]
            merged_shots[-1] = (prev_start, end)
        else:
            merged_shots.append((start, end))
    
    return merged_shots


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
        try:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to save memory and time
            if frame_count % MOTION_FRAME_SKIP != 0:
                frame_count += 1
                continue

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
            
            # Explicitly clear flow to help GC
            del flow
            
        except cv2.error as e:
            print(f"      Warning: OpenCV error during motion extraction at frame {frame_count}: {e}")
            frame_count += 1
            continue
        except Exception as e:
            print(f"      Warning: Unexpected error during motion extraction: {e}")
            break

    cap.release()
    return float(np.mean(scores)) if len(scores) > 0 else 0.0


def compute_motion_scores_array(video_path: Path, start: float, end: float, domain: str) -> np.ndarray:
    """Compute optical flow magnitudes across frames. Returns array for stats (mean, std, peak)."""
    
    if domain in MOTION_DISABLED_DOMAINS:
        return np.array([0.0])

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return np.array([0.0])

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if MOTION_DOWNSCALE_WIDTH and prev_gray.shape[1] > MOTION_DOWNSCALE_WIDTH:
        new_h = int(prev_gray.shape[0] * (MOTION_DOWNSCALE_WIDTH / prev_gray.shape[1]))
        prev_gray = cv2.resize(prev_gray, (MOTION_DOWNSCALE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    scores: List[float] = []
    frame_count = 0
    max_frames = int((end - start) * fps) if fps > 0 else 0

    while frame_count < max_frames:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % MOTION_FRAME_SKIP != 0:
                frame_count += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if MOTION_DOWNSCALE_WIDTH and gray.shape[1] > MOTION_DOWNSCALE_WIDTH:
                new_h = int(gray.shape[0] * (MOTION_DOWNSCALE_WIDTH / gray.shape[1]))
                gray = cv2.resize(gray, (MOTION_DOWNSCALE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

            mag = float(np.mean(np.linalg.norm(flow, axis=2)))
            scores.append(mag)
            prev_gray = gray
            frame_count += 1
            del flow
            
        except (cv2.error, Exception) as e:
            frame_count += 1
            continue

    cap.release()
    return np.array(scores) if scores else np.array([0.0])


def compute_audio_descriptors(audio_path: Path, start: float, end: float) -> Tuple[float, float, float, float, float, float]:
    """Extract rich audio descriptors: RMS energy, delta, spectral flux, pitch mean/std, silence ratio."""
    
    duration = end - start
    if duration <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    try:
        y, sr = librosa.load(str(audio_path), sr=16000, offset=start, duration=duration)
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        rms_energy = float(np.mean(rms))
        rms_delta = float(np.std(rms)) if len(rms) > 1 else 0.0
        
        # Spectral flux (change in spectrum)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spec_flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
        spectral_flux = float(np.mean(spec_flux)) if len(spec_flux) > 0 else 0.0
        
        # Pitch estimation via fundamental frequency
        f0 = librosa.yin(y, fmin=50, fmax=400)
        pitch_mean = float(np.nanmean(f0)) if not np.all(np.isnan(f0)) else 0.0
        pitch_std = float(np.nanstd(f0)) if not np.all(np.isnan(f0)) else 0.0
        
        # Silence ratio
        silence_threshold = 0.02
        silent_frames = np.sum(rms < silence_threshold)
        silence_ratio = float(silent_frames / len(rms)) if len(rms) > 0 else 0.0
        
        return rms_energy, rms_delta, spectral_flux, pitch_mean, pitch_std, silence_ratio
        
    except Exception as e:
        print(f"      Warning: Audio descriptor extraction failed - {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def compute_visual_change_features(video_path: Path, start: float, end: float) -> Tuple[float, float]:
    """Extract visual change features: scene cut strength and color histogram delta."""
    
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return 0.0, 0.0

    # Get last frame
    max_frames = int((end - start) * fps)
    frame_count = 0
    last_frame = first_frame
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
        frame_count += 1

    cap.release()

    # Scene cut strength: compare first and last frame
    diff = cv2.absdiff(first_frame, last_frame)
    scene_cut_strength = float(np.mean(diff) / 255.0)

    # Color histogram delta
    hist_first = cv2.calcHist([first_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_last = cv2.calcHist([last_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    hist_first = cv2.normalize(hist_first, hist_first).flatten()
    hist_last = cv2.normalize(hist_last, hist_last).flatten()
    
    color_hist_delta = float(cv2.compareHist(hist_first, hist_last, cv2.HISTCMP_BHATTACHARYYA))

    return scene_cut_strength, color_hist_delta


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


def compute_audio_peakiness(audio_path: Path, start: float, end: float, window: float = 5.0) -> float:
    """
    Measures how much louder this shot is compared to its local neighborhood.
    """
    y, sr = librosa.load(str(audio_path), sr=16000)
    t_start = int(start * sr)
    t_end = int(end * sr)

    if t_end <= t_start:
        return 0.0

    shot_energy = np.mean(np.abs(y[t_start:t_end]))

    w = int(window * sr)
    local_start = max(0, t_start - w)
    local_end = min(len(y), t_end + w)

    local_energy = np.mean(np.abs(y[local_start:local_end])) + 1e-6
    return float(shot_energy / local_energy)


def extract_shot_features(video_path: Path, audio_path: Path, start: float, end: float, domain: str, shot_index: int = 0, total_shots: int = 1) -> ShotFeatures:
    """Extract rich shot-level descriptors. Computes is_silent flag for model robustness."""
    
    duration = end - start
    relative_position = shot_index / max(total_shots, 1)
    
    # Motion features (mean, std, peak)
    motion_scores = compute_motion_scores_array(video_path, start, end, domain)
    motion_mean = float(np.mean(motion_scores)) if len(motion_scores) > 0 else 0.0
    motion_std = float(np.std(motion_scores)) if len(motion_scores) > 0 else 0.0
    motion_peak = float(np.max(motion_scores)) if len(motion_scores) > 0 else 0.0
    
    # Audio features (rich descriptors)
    rms_energy, rms_delta, spectral_flux, pitch_mean, pitch_std, silence_ratio = compute_audio_descriptors(audio_path, start, end)
    
    # Visual change features
    scene_cut_strength, color_hist_delta = compute_visual_change_features(video_path, start, end)
    
    # Explicit silence flag: helps model ignore noise-only or silent shots
    # Threshold: >90% silence OR RMS energy below 0.01
    is_silent = silence_ratio > 0.9 or rms_energy < 0.01
    
    return ShotFeatures(
        duration=duration,
        relative_position=relative_position,
        motion_mean=motion_mean,
        motion_std=motion_std,
        motion_peak=motion_peak,
        rms_energy=rms_energy,
        rms_delta=rms_delta,
        spectral_flux=spectral_flux,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        silence_ratio=silence_ratio,
        scene_cut_strength=scene_cut_strength,
        color_hist_delta=color_hist_delta,
        is_silent=is_silent,
    )


# ============================================================================
# Step 4: Importance Scoring with Temporal Smoothing
# ============================================================================


def normalize_features_dataset(dataset: List[VideoDataset]) -> List[VideoDataset]:
    """
    Normalize features across entire dataset (not per-video) to maintain cross-video consistency.
    
    Strategy:
      - motion_mean, motion_std, motion_peak: log(1+x) then z-score
      - spectral_flux: log(1+x) then z-score
      - rms_energy, rms_delta: z-score
      - scene_cut_strength, color_hist_delta: z-score
      - silence_ratio, relative_position: keep as-is (already bounded 0-1)
      - is_silent: keep as-is (binary)
    """
    if not dataset:
        return dataset
    
    # Collect all features across dataset
    all_features = {}
    for video in dataset:
        for shot in video.shots:
            feat = shot.features
            for key in ['motion_mean', 'motion_std', 'motion_peak', 'spectral_flux', 
                       'rms_energy', 'rms_delta', 'scene_cut_strength', 'color_hist_delta']:
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(getattr(feat, key))
    
    # Compute statistics for z-score normalization
    stats = {}
    for key, values in all_features.items():
        arr = np.array(values, dtype=float)
        if key in ['motion_mean', 'motion_std', 'motion_peak', 'spectral_flux']:
            # Log transform first (handles wide range and zeros)
            arr = np.log1p(arr)
        stats[key] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr)) + 1e-8}
    
    # Apply normalization
    for video in dataset:
        for shot in video.shots:
            feat = shot.features
            # Normalize motion features
            for key in ['motion_mean', 'motion_std', 'motion_peak']:
                val = getattr(feat, key)
                val_log = np.log1p(val)
                val_norm = (val_log - stats[key]['mean']) / stats[key]['std']
                setattr(feat, key, float(val_norm))
            
            # Normalize spectral flux
            val = feat.spectral_flux
            val_log = np.log1p(val)
            val_norm = (val_log - stats['spectral_flux']['mean']) / stats['spectral_flux']['std']
            feat.spectral_flux = float(val_norm)
            
            # Normalize RMS features
            for key in ['rms_energy', 'rms_delta']:
                val = getattr(feat, key)
                val_norm = (val - stats[key]['mean']) / stats[key]['std']
                setattr(feat, key, float(val_norm))
            
            # Normalize visual features
            for key in ['scene_cut_strength', 'color_hist_delta']:
                val = getattr(feat, key)
                val_norm = (val - stats[key]['mean']) / stats[key]['std']
                setattr(feat, key, float(val_norm))
    
    return dataset


def compute_importance_DEPRECATED(features, weights: Optional[Dict[str, float]] = None, domain: Optional[str] = None) -> float:
    """DEPRECATED: Do not use. Features are extracted; GNN learns importance."""
    raise NotImplementedError(
        "compute_importance has been removed. "
        "Extract features and let the GNN learn importance from them. "
        "Use pseudo-labels for YouTube data: Top-15% of shots by duration or other criteria."
    )


def emphasize_peaks_DEPRECATED(scores: List[float], gamma: float = 2.0) -> List[float]:
    """DEPRECATED: Do not use. Let GNN learn what matters."""
    raise NotImplementedError("emphasize_peaks_DEPRECATED: Let the GNN learn peak importance.")


def smooth_importance_scores_DEPRECATED(scores: List[float], sigma: float = 2.0) -> List[float]:
    """DEPRECATED: Do not use. GNN can learn temporal patterns."""
    raise NotImplementedError("smooth_importance_scores_DEPRECATED: Let the GNN learn temporal patterns.")


# ============================================================================
# Step 5: Full Pipeline
# ============================================================================


def process_video(video_path: Path, audio_path: Path, domain: str = "default", smooth_sigma: float = 2.0) -> VideoDataset:
    """Process video: extract rich features. NO importance scoring. GNN learns from features."""

    shot_boundaries = detect_shots(video_path)
    print(f"Detected {len(shot_boundaries)} shots")

    shots_features: List[ShotFeatures] = []
    for i, (start, end) in enumerate(shot_boundaries):
        print(f"Processing shot {i + 1}/{len(shot_boundaries)}...")
        features = extract_shot_features(video_path, audio_path, start, end, domain, shot_index=i, total_shots=len(shot_boundaries))
        shots_features.append(features)
        if (i + 1) % SHOT_GC_INTERVAL == 0:
            clear_memory()

    shots: List[Shot] = []
    for (start, end), feat in zip(shot_boundaries, shots_features):
        transcription = None
        asr_confidence = None
        if ENABLE_ASR and audio_path is not None:
            # Flip ENABLE_ASR to True to run per-shot Whisper transcription.
            transcription, asr_confidence = transcribe_segment(audio_path, start, end)

        shots.append(
            Shot(
                start=start,
                end=end,
                features=feat,
                label=None,
                transcription=transcription,
                asr_confidence=asr_confidence,
            )
        )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()
    clear_memory()

    return VideoDataset(video_id=video_path.stem, duration=duration, domain=domain, shots=shots)


def apply_pseudo_labels_youtube(dataset: List[VideoDataset], top_k_ratio: float = 0.15) -> List[VideoDataset]:
    """
    Apply pseudo-labels to YouTube videos: Top-K shots per video → label 1, rest → label 0.
    
    Args:
        dataset: List of VideoDataset objects
        top_k_ratio: Fraction of shots to label as 1 (default 15%)
    
    Returns:
        Updated dataset with labels assigned
    """
    for video in dataset:
        num_shots = len(video.shots)
        if num_shots == 0:
            continue
        
        # Calculate top-K count
        top_k = max(1, int(np.ceil(num_shots * top_k_ratio)))
        
        # Score shots by duration as a simple heuristic for pseudo-labels
        # (GNN will learn better patterns, but duration is a reasonable proxy)
        durations = [s.end - s.start for s in video.shots]
        top_k_indices = np.argsort(durations)[-top_k:].tolist()
        
        # Assign labels
        for i, shot in enumerate(video.shots):
            shot.label = 1.0 if i in top_k_indices else 0.0
    
    return dataset


def build_dataset(
    video_dir: Path = RAW_YOUTUBE_DIR,
    output_path: Optional[Path] = None,
    max_per_playlist: Optional[int] = 10,
    batch_size: int = 3,
    domain_map: Optional[Dict[str, Dict[str, str]]] = None,
    playlist_filter: Optional[List[str]] = None,
    save_frequency: str = "playlist",
) -> List[VideoDataset]:
    video_root = Path(video_dir)
    dataset: List[VideoDataset] = []

    if output_path is None:
        output_path = UNIFIED_OUTPUT_DIR / "youtube" / "complete_dataset.json"

    if not video_root.exists():
        print(f"⚠️  Video root not found: {video_root}")
        return dataset

    # Load already-processed video IDs to skip them
    processed_ids = get_processed_video_ids("unified")
    if processed_ids:
        print(f"✓ Found {len(processed_ids)} already-processed videos, will skip them")

    playlist_dirs = [p for p in video_root.iterdir() if p.is_dir()]
    if playlist_filter:
        playlist_dirs = [p for p in playlist_dirs if p.name in playlist_filter]
    
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

        # Filter out already-processed videos
        video_paths = [v for v in video_paths if v.stem not in processed_ids]
        if not video_paths:
            print(f"⏭️  All videos in {playlist_name} already processed")
            continue

        print(f"\n{'='*60}")
        print(f"Playlist: {playlist_name} ({domain})")
        print(f"Found {len(video_paths)} videos (processing up to {len(video_paths)})")

        for batch in chunked(video_paths, batch_size):
            for video_path in batch:
                # Skip if already processed
                if video_path.stem in processed_ids:
                    print(f"  ⏭️  {video_path.stem} already processed")
                    continue

                print(f"  Processing {video_path.stem}...")
                try:
                    audio_path = extract_audio(video_path)
                    if not audio_path:
                        print(f"    ✗ No audio found for {video_path.stem}")
                        continue
                    
                    video_data = process_video(video_path, audio_path, domain=domain)
                    dataset.append(video_data)
                    print(f"    ✓ Extracted {len(video_data.shots)} shots")
                    if save_frequency == "video":
                        save_unified_dataset_incremental(dataset)
                except Exception as exc:
                    print(f"    ✗ Error processing {video_path.stem}: {exc}")
                    continue
                finally:
                    clear_memory()
            clear_memory()

        if save_frequency in {"playlist", "video"} and dataset:
            save_unified_dataset_incremental(dataset)

    # Apply pseudo-labels to YouTube videos before saving
    if dataset:
        dataset = apply_pseudo_labels_youtube(dataset, top_k_ratio=0.15)
    
    # Normalize features across dataset (fixes scale imbalance issue)
    if dataset:
        print("\n📊 Normalizing features across dataset...")
        dataset = normalize_features_dataset(dataset)
        print("✓ Feature normalization complete (log+zscore applied)")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([v.to_dict() for v in dataset], f, indent=2)
        print(f"\n✓ Dataset saved to {output_path}")
        print(f"Total videos: {len(dataset)}")
        print(f"Total shots: {sum(len(v.shots) for v in dataset)}")

    return dataset


# ============================================================================
# TVSum & SumMe Dataset Loaders
# ============================================================================


def load_tvsumme_dataset(json_file: Path, video_path: Optional[Path] = None, audio_path: Optional[Path] = None, domain: str = "tvsum") -> Optional[List[VideoDataset]]:
    """
    Load TVSum/SumMe JSON with ground-truth importance scores.
    Extract 14D features from video/audio if provided, otherwise use defaults.
    Normalize per-video and assign as labels.
    
    Expected format:
    {
      "video_id": "...",
      "duration": 300,
      "shots": [
        {"start": 0, "end": 1.5, "user_summary": [0, 1, 0, ...], "features": {...}},
        ...
      ]
    }
    """
    try:
        with open(json_file) as f:
            video_data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None

    shots_list = video_data.get("shots", [])
    if not shots_list:
        return None

    # Extract importance scores from user summaries
    importance_scores = []
    for shot in shots_list:
        user_summary = shot.get("user_summary", [])
        if user_summary:
            importance = float(np.mean(user_summary))
        else:
            importance = shot.get("importance", 0.0)
        importance_scores.append(importance)

    # Normalize per-video
    scores = np.array(importance_scores)
    if scores.max() - scores.min() > 1e-6:
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores_norm = scores

    # Determine if we can extract features from video
    can_extract_features = video_path is not None and audio_path is not None and video_path.exists() and audio_path.exists()
    
    if can_extract_features:
        print(f"    Extracting 14D features from {video_path.stem}...")
    else:
        if video_path or audio_path:
            print(f"    Warning: Video or audio path missing/invalid, using default features")

    # Build shots with extracted or default features
    shots = []
    for i, (shot_data, label) in enumerate(zip(shots_list, scores_norm)):
        start = float(shot_data.get("start", 0))
        end = float(shot_data.get("end", 1))
        
        # Try to extract real features from video/audio
        if can_extract_features:
            try:
                features = extract_shot_features(
                    video_path, audio_path, start, end, domain,
                    shot_index=i, total_shots=len(shots_list)
                )
            except Exception as e:
                print(f"      Warning: Feature extraction failed for shot {i}: {e}. Using defaults.")
                features = ShotFeatures(
                    duration=end - start,
                    relative_position=i / max(len(shots_list), 1),
                    motion_mean=0.0,
                    motion_std=0.0,
                    motion_peak=0.0,
                    rms_energy=0.0,
                    rms_delta=0.0,
                    spectral_flux=0.0,
                    pitch_mean=0.0,
                    pitch_std=0.0,
                    silence_ratio=0.0,
                    scene_cut_strength=0.0,
                    color_hist_delta=0.0,
                )
        else:
            # Use default features
            features = ShotFeatures(
                duration=end - start,
                relative_position=i / max(len(shots_list), 1),
                motion_mean=0.0,
                motion_std=0.0,
                motion_peak=0.0,
                rms_energy=0.0,
                rms_delta=0.0,
                spectral_flux=0.0,
                pitch_mean=0.0,
                pitch_std=0.0,
                silence_ratio=0.0,
                scene_cut_strength=0.0,
                color_hist_delta=0.0,
            )
        
        shots.append(Shot(start=start, end=end, features=features, label=float(label)))

    duration = float(video_data.get("duration", shots_list[-1].get("end", 0) if shots_list else 0))
    video_id = video_data.get("video_id", json_file.stem)

    dataset = VideoDataset(video_id=video_id, duration=duration, domain=domain, shots=shots)
    return [dataset]


def preprocess_tvsumme_videos(dataset_name: str = "tvsum") -> None:
    """Preprocess TVSum/SumMe videos into JSON files with features."""
    
    # Set directories
    if dataset_name == "tvsum":
        video_dir = Path("model/data/raw/tvsum/video")
        output_dir = RAW_TVSUMME_DIR
    else:  # summe
        video_dir = Path("model/data/raw/summe/videos")
        output_dir = RAW_SUMME_DIR
    
    if not video_dir.exists():
        print(f"⚠️  Video directory not found: {video_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all video files
    video_files = list(video_dir.glob("*.mp4"))
    if not video_files:
        print(f"⚠️  No MP4 files found in {video_dir}")
        return
    
    print(f"\n🎬 Preprocessing {len(video_files)} {dataset_name.upper()} videos...")
    
    for video_path in video_files:
        video_id = video_path.stem
        json_file = output_dir / f"{video_id}.json"
        
        # Skip if already processed
        if json_file.exists():
            print(f"   ⏭️  {video_id} (already processed)")
            continue
        
        print(f"\n   Processing {video_id}...")
        
        try:
            # Extract audio
            audio_path = extract_audio(video_path, output_dir=str(output_dir / "audio"))
            if not audio_path:
                print(f"      ⚠️  No audio stream found")
                continue
            
            # Process video to extract features
            video_data = process_video(video_path, audio_path, domain=dataset_name, smooth_sigma=2.0)
            
            # Apply pseudo-labels (top-15% by duration)
            num_shots = len(video_data.shots)
            top_k = max(1, int(np.ceil(num_shots * 0.15)))
            durations = [s.end - s.start for s in video_data.shots]
            top_k_indices = np.argsort(durations)[-top_k:].tolist()
            
            for i, shot in enumerate(video_data.shots):
                shot.label = 1.0 if i in top_k_indices else 0.0
            
            # Save as JSON
            with open(json_file, "w") as f:
                json.dump(video_data.to_dict(), f, indent=2)
            
            print(f"      ✓ Saved {len(video_data.shots)} shots")
            clear_memory()
            
        except Exception as e:
            print(f"      ✗ Error: {e}")
            clear_memory()


def load_all_tvsumme(dataset_name: str = "tvsum") -> List[VideoDataset]:
    """Load all TVSum or SumMe videos from preprocessed JSON files."""
    raw_dir = RAW_TVSUMME_DIR if dataset_name == "tvsum" else RAW_SUMME_DIR
    
    # First, preprocess if needed
    preprocess_tvsumme_videos(dataset_name)
    
    if not raw_dir.exists():
        print(f"⚠️  {dataset_name.upper()} directory not found: {raw_dir}")
        return []

    all_videos = []
    json_files = sorted(raw_dir.glob("*.json"))
    
    if not json_files:
        print(f"⚠️  No JSON files found in {raw_dir} after preprocessing")
        return []

    print(f"\nLoading {len(json_files)} {dataset_name.upper()} videos from preprocessed JSON...")
    
    for json_file in json_files:
        video_id = json_file.stem
        
        try:
            with open(json_file) as f:
                video_dict = json.load(f)
            
            # Reconstruct VideoDataset from JSON
            shots = [
                Shot(
                    start=s["start"],
                    end=s["end"],
                    features=ShotFeatures(**s["features"]),
                    label=s.get("label"),
                    rank=s.get("rank")
                )
                for s in video_dict.get("shots", [])
            ]
            
            video = VideoDataset(
                video_id=video_dict["video_id"],
                duration=video_dict["duration"],
                domain=dataset_name,
                shots=shots
            )
            all_videos.append(video)
            print(f"   ✓ {video_id}: {len(shots)} shots, {video.duration:.1f}s")
            
        except Exception as e:
            print(f"   ✗ Error loading {video_id}: {e}")
        
        clear_memory()

    return all_videos


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
    "BBC-Breaking-News": {
        "url": "https://www.youtube.com/playlist?list=PLS3XGZxi7cBVTzEE4Sim9UuNKnUJq9Vkh",
        "domain": "documentary",
        "description": "News and current events",
    },
    "ESPN-Highlights": {
        "url": "https://www.youtube.com/playlist?list=PL87LlAF-2PIwKpIUaKO4_p5QNmjxhYUFG",
        "domain": "sports",
        "description": "Sports highlights",
    },
    "BBC-Learning": {
        "url": "https://www.youtube.com/playlist?list=PLcetZ6gSk969oGvAI0e4_PgVnlGbm64bp",
        "domain": "documentary",
        "description": "BBC educational content",
    },
}

# ============================================================================
# High-Importance Curated Videos
# ============================================================================

HIGH_IMPORTANCE_VIDEOS: Dict[str, Dict[str, str]] = {
    "sNPnbI1arSE": {
        "url": "https://youtu.be/sNPnbI1arSE",
        "domain": "music",
        "description": "Eminem - My Name Is (Official Music Video) | 4 min music video with high visual complexity",
    },
    "XbGs_qK2PQA": {
        "url": "https://youtu.be/XbGs_qK2PQA",
        "domain": "music",
        "description": "Eminem - Rap God (Explicit) | 6 min high-motion music video with fast cuts",
    },
    "VQRLujxTm3c": {
        "url": "https://youtu.be/VQRLujxTm3c",
        "domain": "movie_trailer",
        "description": "Grand Theft Auto VI Trailer 2 | 3 min cinematic game trailer with dynamic scenes",
    },
    "JfVOs4VSpmA": {
        "url": "https://youtu.be/JfVOs4VSpmA",
        "domain": "movie_trailer",
        "description": "SPIDER-MAN: NO WAY HOME - Official Trailer (HD) | 3 min action movie trailer",
    },
    "yzr2rXRGJz8": {
        "url": "https://youtu.be/yzr2rXRGJz8",
        "domain": "sports",
        "description": "Siraj Takes 5fer in UNBELIEVABLE Finish! | England v India Test Highlights | 6.5 min cricket highlights",
    },
    "JIRqdeNl2cU": {
        "url": "https://youtu.be/JIRqdeNl2cU",
        "domain": "sports",
        "description": "Race Highlights | 2025 Dutch Grand Prix | 8 min F1 race highlights with high-speed action",
    },
    "s7WPMv2IgFk": {
        "url": "https://youtu.be/s7WPMv2IgFk",
        "domain": "movie_clip",
        "description": "The Taliban Attack US Marines Scene | JARHEAD 2: FIELD OF FIRE (2014) | 5 min action movie clip",
    },
    "oq02OFHhTSE": {
        "url": "https://youtu.be/oq02OFHhTSE",
        "domain": "gaming",
        "description": "low sens = god aim (CSGO2 edit) | 2.5 min competitive gaming montage with quick reflexes",
    },
    "8ZfBKk7YneU": {
        "url": "https://youtu.be/8ZfBKk7YneU",
        "domain": "gaming",
        "description": "kyousuke - DREAM SPACE CSGO2 edit | 3 min gaming highlight reel with visual effects",
    },
    # Additional music videos
    "4NRXx6U8ABQ": {
        "url": "https://youtu.be/4NRXx6U8ABQ",
        "domain": "music",
        "description": "Music video with high visual complexity and dynamic scenes",
    },
    "WWEs82u37Mw": {
        "url": "https://youtu.be/WWEs82u37Mw",
        "domain": "music",
        "description": "Music video with fast cuts and motion",
    },
    "_CL6n0FJZpk": {
        "url": "https://youtu.be/_CL6n0FJZpk",
        "domain": "music",
        "description": "Music video with cinematic elements",
    },
    "H5v3kku4y6Q": {
        "url": "https://youtu.be/H5v3kku4y6Q",
        "domain": "music",
        "description": "Music video with choreography and effects",
    },
    "suAR1PYFNYA": {
        "url": "https://youtu.be/suAR1PYFNYA",
        "domain": "music",
        "description": "Music video with narrative storytelling",
    },
    "TUVcZfQe-Kw": {
        "url": "https://youtu.be/TUVcZfQe-Kw",
        "domain": "music",
        "description": "Music video with performance elements",
    },
    # Additional movie trailers
    "73_1biulkYk": {
        "url": "https://youtu.be/73_1biulkYk",
        "domain": "movie_trailer",
        "description": "Movie trailer with action sequences",
    },
    "XJMuhwVlca4": {
        "url": "https://youtu.be/XJMuhwVlca4",
        "domain": "movie_trailer",
        "description": "Movie trailer with dramatic scenes",
    },
    "j7jPnwVGdZ8": {
        "url": "https://youtu.be/j7jPnwVGdZ8",
        "domain": "movie_trailer",
        "description": "Movie trailer with fast-paced editing",
    },
    "aDyQxtg0V2w": {
        "url": "https://youtu.be/aDyQxtg0V2w",
        "domain": "movie_trailer",
        "description": "Movie trailer with cinematic visuals",
    },
    "hRFY_Fesa9Q": {
        "url": "https://youtu.be/hRFY_Fesa9Q",
        "domain": "movie_trailer",
        "description": "Movie trailer with suspense and action",
    },
    "SzINZZ6iqxY": {
        "url": "https://youtu.be/SzINZZ6iqxY",
        "domain": "movie_trailer",
        "description": "Movie trailer with dynamic scene transitions",
    },
    # Additional gaming montages
    "IKYnbxX8ujk": {
        "url": "https://youtu.be/IKYnbxX8ujk",
        "domain": "gaming",
        "description": "Gaming montage with competitive gameplay",
    },
    "a8keyU_0kJ4": {
        "url": "https://youtu.be/a8keyU_0kJ4",
        "domain": "gaming",
        "description": "Gaming montage with skill highlights",
    },
    "Z6qUyGjFOEE": {
        "url": "https://youtu.be/Z6qUyGjFOEE",
        "domain": "gaming",
        "description": "Gaming montage with fast-paced action",
    },
    "oyGu3fwxDwk": {
        "url": "https://youtu.be/oyGu3fwxDwk",
        "domain": "gaming",
        "description": "Gaming montage with clutch moments",
    },
    "CoRiY1rSCCA": {
        "url": "https://youtu.be/CoRiY1rSCCA",
        "domain": "gaming",
        "description": "Gaming montage with visual effects",
    },
    "fvSzkKLw4pE": {
        "url": "https://youtu.be/fvSzkKLw4pE",
        "domain": "gaming",
        "description": "Gaming montage with synchronized edits",
    },
    # Additional sports highlights
    "hGmAPvLuBOQ": {
        "url": "https://youtu.be/hGmAPvLuBOQ",
        "domain": "sports",
        "description": "Sports highlights with key moments",
    },
    "amMJfaB5dXo": {
        "url": "https://youtu.be/amMJfaB5dXo",
        "domain": "sports",
        "description": "Sports highlights with high-speed action",
    },
    "EPwOPr2xkYo": {
        "url": "https://youtu.be/EPwOPr2xkYo",
        "domain": "sports",
        "description": "Sports highlights with dramatic plays",
    },
    "TthnLjCrMTg": {
        "url": "https://youtu.be/TthnLjCrMTg",
        "domain": "sports",
        "description": "Sports highlights with intense competition",
    },
    "ZW_YxG7iz8c": {
        "url": "https://youtu.be/ZW_YxG7iz8c",
        "domain": "sports",
        "description": "Sports highlights with crowd reactions",
    },
    "KiOAjVm5wug": {
        "url": "https://youtu.be/KiOAjVm5wug",
        "domain": "sports",
        "description": "Sports highlights with game-winning moments",
    },
}

# ============================================================================
# Dataset Management & Testing
# ============================================================================


def validate_dataset(dataset: List[VideoDataset]) -> Dict[str, object]:
    if not dataset:
        return {"valid": False, "error": "Empty dataset"}

    all_shots = [s for v in dataset for s in v.shots]
    labeled_shots = [s for s in all_shots if s.label is not None]

    stats = {
        "valid": True,
        "num_videos": len(dataset),
        "total_shots": len(all_shots),
        "labeled_shots": len(labeled_shots),
        "total_duration": sum(v.duration for v in dataset),
        "avg_shots_per_video": len(all_shots) / len(dataset) if dataset else 0,
        "avg_duration_per_video": sum(v.duration for v in dataset) / len(dataset) if dataset else 0,
        "domains": list(set(v.domain for v in dataset)),
        "label_stats": {
            "labeled": len(labeled_shots),
            "unlabeled": len(all_shots) - len(labeled_shots),
            "label_distribution": {
                "0": len([s for s in labeled_shots if s.label == 0.0]),
                "1": len([s for s in labeled_shots if s.label == 1.0]),
            } if labeled_shots else {},
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


def save_dataset_structure(dataset: List[VideoDataset], output_dir: Path = None) -> Dict[str, Path]:
    if output_dir is None:
        output_dir = UNIFIED_OUTPUT_DIR / "youtube"
    
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
    
    # Create domain-specific subdirectories
    domains_in_dataset = set(v.domain for v in dataset)
    for domain in domains_in_dataset:
        (features_dir / domain).mkdir(exist_ok=True)
        (metadata_dir / domain).mkdir(exist_ok=True)

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

    # Save per-domain metadata summaries
    for domain in domains_in_dataset:
        domain_videos = [v for v in metadata_videos if v["domain"] == domain]
        domain_metadata = {
            "domain": domain,
            "num_videos": len(domain_videos),
            "videos": domain_videos,
            "total_shots": sum(v["num_shots"] for v in domain_videos),
        }
        domain_metadata_file = metadata_dir / domain / "domain_metadata.json"
        with open(domain_metadata_file, "w") as f:
            json.dump(domain_metadata, f, indent=2)

    for video in dataset:
        video_features = {
            "video_id": video.video_id,
            "duration": video.duration,
            "domain": video.domain,
            "shots": [],
        }
        for shot in video.shots:
            video_features["shots"].append(
                {
                    "start": shot.start,
                    "end": shot.end,
                    "label": shot.label,
                    "rank": shot.rank,
                    "features": shot.features.to_dict(),
                }
            )

        features_file = features_dir / video.domain / f"{video.video_id}_features.json"
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


def get_processed_video_ids(dataset_type: str = "unified") -> set:
    """Get set of already-processed video IDs to avoid re-processing."""
    if dataset_type == "unified":
        combined_file = UNIFIED_OUTPUT_DIR / "combined" / "all_videos.json"
        if not combined_file.exists():
            return set()
        try:
            with open(combined_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return {v["video_id"] for v in data}
                else:
                    return {v["video_id"] for v in data.get("videos", [])}
        except Exception:
            return set()
    return set()


def load_existing_unified_dataset() -> List[VideoDataset]:
    """Load existing unified dataset if it exists."""
    combined_file = UNIFIED_OUTPUT_DIR / "combined" / "all_videos.json"
    
    if not combined_file.exists():
        return []
    
    try:
        with open(combined_file) as f:
            data = json.load(f)
        
        videos = []
        for v_dict in data:
            shots = [
                Shot(
                    start=s["start"],
                    end=s["end"],
                    features=ShotFeatures(**s["features"]),
                    label=s.get("label"),
                    rank=s.get("rank")
                )
                for s in v_dict["shots"]
            ]
            videos.append(VideoDataset(
                video_id=v_dict["video_id"],
                duration=v_dict["duration"],
                domain=v_dict["domain"],
                shots=shots
            ))
        return videos
    except Exception as e:
        print(f"Warning: Could not load existing unified dataset: {e}")
        return []


def save_unified_dataset_incremental(all_videos: List[VideoDataset], save_every: int = 10) -> Dict[str, Path]:
    """
    Save unified dataset incrementally. Every 10 videos, merge into combined file.
    Handles resumption - skips duplicates.
    """
    unified_root = UNIFIED_OUTPUT_DIR
    unified_root.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each source
    for source in ["tvsum", "summe", "youtube", "combined"]:
        source_dir = unified_root / source
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "features").mkdir(exist_ok=True)
        (source_dir / "metadata").mkdir(exist_ok=True)
        (source_dir / "splits").mkdir(exist_ok=True)

    # Load existing unified dataset to avoid duplicates
    existing_videos = load_existing_unified_dataset()
    existing_ids = {v.video_id for v in existing_videos}
    
    # Filter new videos (skip already processed)
    new_videos = [v for v in all_videos if v.video_id not in existing_ids]
    
    if new_videos:
        print(f"\n📝 Saving {len(new_videos)} new videos (skipping {len(existing_videos)} existing)...")
        existing_videos.extend(new_videos)
    else:
        print(f"\n✓ All {len(all_videos)} videos already processed")
        existing_videos = all_videos

    # Save per-source metadata and features
    for source in ["tvsum", "summe", "youtube"]:
        # Map videos to source: tvsum/summe match exactly, all others go to youtube
        if source in ["tvsum", "summe"]:
            source_videos = [v for v in existing_videos if v.domain == source]
        else:  # source == "youtube"
            # All non-tvsum/summe videos are YouTube (music, sports, gaming, etc.)
            source_videos = [v for v in existing_videos if v.domain not in ["tvsum", "summe"]]
        
        if not source_videos:
            continue

        source_dir = unified_root / source
        
        # Save features
        for video in source_videos:
            video_data = {
                "video_id": video.video_id,
                "duration": video.duration,
                "domain": video.domain,
                "shots": [s.to_dict() for s in video.shots]
            }
            features_file = source_dir / "features" / f"{video.video_id}_features.json"
            with open(features_file, "w") as f:
                json.dump(video_data, f, indent=2)

        # Save metadata
        splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        for idx, video in enumerate(source_videos):
            split = _assign_split(idx)
            splits[split].append(video.video_id)

        metadata = {
            "source": source,
            "num_videos": len(source_videos),
            "total_shots": sum(len(v.shots) for v in source_videos),
            "videos": [
                {
                    "video_id": v.video_id,
                    "duration": v.duration,
                    "num_shots": len(v.shots),
                    "label_stats": {
                        "labeled": len([s for s in v.shots if s.label is not None]),
                        "unlabeled": len([s for s in v.shots if s.label is None]),
                    }
                }
                for v in source_videos
            ]
        }
        
        metadata_file = source_dir / "metadata" / "dataset_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        splits_file = source_dir / "splits" / "train_val_test_split.json"
        with open(splits_file, "w") as f:
            json.dump(splits, f, indent=2)

    # Save combined dataset
    combined_dir = unified_root / "combined"
    all_videos_file = combined_dir / "all_videos.json"
    
    with open(all_videos_file, "w") as f:
        json.dump([v.to_dict() for v in existing_videos], f, indent=2)

    # Save combined statistics
    all_shots = [s for v in existing_videos for s in v.shots]
    labeled_shots = [s for s in all_shots if s.label is not None]
    
    stats = {
        "total_videos": len(existing_videos),
        "total_shots": len(all_shots),
        "labeled_shots": len(labeled_shots),
        "by_source": {},
        "label_distribution": {
            "0": len([s for s in labeled_shots if s.label == 0.0]),
            "1": len([s for s in labeled_shots if s.label == 1.0]),
        }
    }

    for source in ["tvsum", "summe", "youtube"]:
        source_videos = [v for v in existing_videos if v.domain == source]
        if source_videos:
            source_shots = [s for v in source_videos for s in v.shots]
            source_labeled = [s for s in source_shots if s.label is not None]
            stats["by_source"][source] = {
                "num_videos": len(source_videos),
                "num_shots": len(source_shots),
                "labeled": len(source_labeled),
                "label_0": len([s for s in source_labeled if s.label == 0.0]),
                "label_1": len([s for s in source_labeled if s.label == 1.0]),
            }

    stats_file = combined_dir / "statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    return {
        "combined_videos": all_videos_file,
        "combined_stats": stats_file,
        "root_dir": unified_root,
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
    all_shots_valid = all(hasattr(s, "start") and hasattr(s, "end") and hasattr(s, "features") for v in dataset for s in v.shots)
    if all_shots_valid:
        print("✓ Test 3: All shots have valid structure")
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
    valid_domains = {"lecture", "interview", "sports", "documentary", "gaming", "music", "movie_trailer", "movie_clip", "default", "tvsum", "summe"}
    all_domains_valid = all(v.domain in valid_domains for v in dataset)
    if all_domains_valid:
        print("✓ Test 5: All videos have valid domain labels")
        tests_passed += 1
    else:
        print("✗ Test 5: Some videos have invalid domain labels")

    tests_total += 1
    all_features_valid = all(hasattr(s.features, "motion_mean") and hasattr(s.features, "rms_energy") for v in dataset for s in v.shots)
    if all_features_valid:
        print("✓ Test 6: All shots have rich feature descriptors")
        tests_passed += 1
    else:
        print("✗ Test 6: Some shots missing feature descriptors")

    tests_total += 1
    stats = validate_dataset(dataset)
    print("✓ Test 7: Dataset statistics computed")
    print(f"   - Videos: {stats['num_videos']}")
    print(f"   - Total shots: {stats['total_shots']}")
    print(f"   - Avg shots/video: {stats['avg_shots_per_video']:.1f}")
    print(f"   - Duration: {stats['total_duration']/3600:.2f}h")
    print(f"   - Labeled: {stats['label_stats']['labeled']} | Unlabeled: {stats['label_stats']['unlabeled']}")
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
    print("📺 UNIFIED VIDEO SUMMARIZATION DATASET BUILDER")
    print("   (TVSum + SumMe + YouTube)")
    print("=" * 70)

    # Unified dataset mode
    UNIFIED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAX_VIDEOS_PER_PLAYLIST = 10

    # Parse command-line arguments
    target_playlists = []
    save_frequency = "playlist"
    skip_high_importance = False
    skip_playlists = False
    skip_tvsumme = False
    
    for arg in sys.argv[1:]:
        if arg == "--include-curated":
            skip_high_importance = False
        elif arg in {"--skip-high-importance", "--skip-curated"}:
            skip_high_importance = True
        elif arg in {"--only-curated", "--curated-only"}:
            skip_high_importance = False
            skip_playlists = True
        elif arg == "--skip-tvsumme":
            skip_tvsumme = True
        elif arg == "--only-tvsumme":
            skip_high_importance = True
            skip_playlists = True
        elif arg in {"playlist", "video"}:
            save_frequency = arg
        elif not arg.startswith("--"):
            target_playlists.append(arg)
    
    if target_playlists and "--include-curated" not in sys.argv:
        skip_high_importance = True

    all_datasets = []

    # Load TVSum & SumMe (if not skipped)
    if not skip_tvsumme:
        print("\n✓ Loading TVSum & SumMe datasets...\n")
        all_datasets.extend(load_all_tvsumme("tvsum"))
        all_datasets.extend(load_all_tvsumme("summe"))
        
        # Save TVSum/SumMe to unified dataset
        if all_datasets:
            print(f"\n💾 Saving TVSum/SumMe datasets ({len(all_datasets)} videos)...")
            save_unified_dataset_incremental(all_datasets)
            print(f"✓ Saved to unified dataset structure")
    else:
        print("\n⏭️  Skipping TVSum & SumMe datasets\n")
    
    # Process high-importance curated videos (unless skipped)
    if not skip_high_importance:
        print("\n✓ Processing High-Importance Curated Videos\n")
        print(f"{'='*70}")
        print(f"Playlist: HIGH_IMPORTANCE_VIDEOS")
        print(f"Found {len(HIGH_IMPORTANCE_VIDEOS)} videos")
        print(f"{'='*70}\n")
    
        for video_id, video_info in HIGH_IMPORTANCE_VIDEOS.items():
            video_url = video_info["url"]
            domain = video_info["domain"]
            
            # Skip if already in unified dataset
            existing_ids = {v.video_id for v in all_datasets}
            if video_id in existing_ids:
                print(f"   Skipping {video_id} (already processed)")
                continue

            print(f"\nProcessing: {video_id} ({domain})")
            
            try:
                video_path = download_individual_video(video_url, video_id, RAW_YOUTUBE_DIR)
                if video_path and video_path.exists():
                    audio_path = extract_audio(video_path)
                    if audio_path:
                        video_data = process_video(video_path, audio_path, domain=domain)
                        all_datasets.append(video_data)
                        print(f"✓ Completed: {len(video_data.shots)} shots")
                        
                        # Incremental save after EVERY video to prevent data loss
                        print(f"💾 Saving progress ({len(all_datasets)} videos)...")
                        save_unified_dataset_incremental(all_datasets)
                        print(f"✓ Saved to unified dataset (tvsum/summe/youtube/combined with features/metadata/splits)")
                    else:
                        print(f"⚠️ No audio found in {video_id}")
                else:
                    print(f"✗ Failed to download {video_id}")
            except Exception as exc:
                print(f"✗ Error processing {video_id}: {exc}")
            finally:
                clear_memory()
    else:
        print("\n⏭️  Skipping high-importance curated videos\n")
    
    # Process YouTube playlists (unless skipped)
    if not skip_playlists:
        print("\n✓ Processing YouTube Playlists\n")
        youtube_datasets = build_dataset(
            video_dir=RAW_YOUTUBE_DIR,
            output_path=None,
            max_per_playlist=MAX_VIDEOS_PER_PLAYLIST,
            batch_size=3,
            domain_map=TOP_5_PLAYLISTS,
            playlist_filter=target_playlists if target_playlists else None,
            save_frequency=save_frequency,
        )
        all_datasets.extend(youtube_datasets)
    else:
        print("\n✓ Skipping YouTube Playlists (processing curated videos only)\n")

    if all_datasets:
        print("\n" + "=" * 70)
        print("FINAL DATASET VALIDATION & UNIFIED STORAGE")
        print("=" * 70)

        is_valid = test_dataset(all_datasets)

        if is_valid:
            print("\n💾 Saving unified dataset...")
            
            # Apply pseudo-labels to YouTube if not already labeled
            for video in all_datasets:
                if video.domain == "youtube":
                    for shot in video.shots:
                        if shot.label is None:
                            shot.label = 0.0
            
            # Save with incremental/merge approach
            saved_paths = save_unified_dataset_incremental(all_datasets)

            print("\n✓ Unified dataset saved successfully!")
            print(f"   Root: {saved_paths['root_dir']}")
            print(f"   Combined videos: {saved_paths['combined_videos']}")
            print(f"   Statistics: {saved_paths['combined_stats']}")

            # Load and display final statistics
            with open(saved_paths['combined_stats']) as f:
                stats = json.load(f)

            print("\n" + "=" * 70)
            print("📊 UNIFIED DATASET STATISTICS")
            print("=" * 70)
            print(f"Total videos: {stats['total_videos']}")
            print(f"Total shots: {stats['total_shots']}")
            print(f"Total labeled: {stats['labeled_shots']}")
            
            print(f"\nBy source:")
            for source in ["tvsum", "summe", "youtube"]:
                if source in stats["by_source"]:
                    src_stats = stats["by_source"][source]
                    print(f"  {source.upper():<10} Videos: {src_stats['num_videos']:>4} | Shots: {src_stats['num_shots']:>5} | Labeled: {src_stats['labeled']:>5} | Label 0: {src_stats['label_0']:>5} | Label 1: {src_stats['label_1']:>5}")
            
            print(f"\nLabel distribution (all):")
            print(f"  Label 0 (negative): {stats['label_distribution']['0']}")
            print(f"  Label 1 (positive): {stats['label_distribution']['1']}")
            print("=" * 70 + "\n")
        else:
            print("\n✗ Dataset validation failed!")
            sys.exit(1)
    else:
        print("\n⚠️  No datasets were successfully created!")
        print("\nUsage:")
        print("  python youtube_dataset.py                                    # TVSum + SumMe + all playlists")
        print("  python youtube_dataset.py --skip-tvsumme                    # YouTube only")
        print("  python youtube_dataset.py --only-tvsumme                    # TVSum + SumMe only")
        print("  python youtube_dataset.py --only-curated                    # Curated YouTube only")
        print("  python youtube_dataset.py TED-Talks --include-curated       # Curated + specific playlist")
        print("\n  (Re-run at any time to resume from checkpoint)")
        print("\nOutput structure:")
        print("  model/data/processed/unified_dataset/")
        print("  ├── tvsum/          (TVSum features)")
        print("  ├── summe/          (SumMe features)")
        print("  ├── youtube/        (YouTube features)")
        print("  └── combined/       (Merged dataset + statistics)")


