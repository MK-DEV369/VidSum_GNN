"""
Handcrafted feature extraction for shots.
Extracts 14-dimensional feature vectors used during GNN training.
"""
import torch
import numpy as np
from typing import List, Tuple
from vidsum_gnn.utils.logging import get_logger

logger = get_logger(__name__)


def extract_handcrafted_features(
    shots_times: List[Tuple[float, float]],
    video_duration: float
) -> torch.Tensor:
    """
    Extract 14-dimensional handcrafted features for each shot.
    
    Features (same order as training):
    1. duration - Shot duration in seconds
    2. relative_position - Position in video (0-1)
    3-5. motion_mean, motion_std, motion_peak - Motion statistics (placeholder)
    6-7. rms_energy, rms_delta - Audio energy (placeholder)
    8. spectral_flux - Audio spectral change (placeholder)
    9-10. pitch_mean, pitch_std - Pitch statistics (placeholder)
    11. silence_ratio - Silent frames ratio (placeholder)
    12. scene_cut_strength - Scene transition strength (placeholder)
    13. color_hist_delta - Color histogram change (placeholder)
    14. is_silent - Binary silent flag (placeholder)
    
    Note: For inference, many features are placeholders (zeros) since they
    require expensive frame-level analysis. The GNN was trained with these
    but they contribute minimally compared to deep embeddings.
    
    Args:
        shots_times: List of (start_sec, end_sec) tuples
        video_duration: Total video duration in seconds
    
    Returns:
        Tensor of shape (N, 14) with handcrafted features
    """
    num_shots = len(shots_times)
    features = []

    if num_shots == 0:
        logger.warning("No shots provided for handcrafted features")
        return torch.empty(0, 14, dtype=torch.float32)

    # Quick stats for debugging shot detection/segmentation behavior
    try:
        durs = [float(e - s) for s, e in shots_times if (e - s) > 0]
        if durs:
            logger.info(
                f"Handcrafted inputs: shots={num_shots}, duration={float(video_duration):.2f}s, "
                f"shot_dur_min={min(durs):.2f}s, shot_dur_med={float(np.median(durs)):.2f}s, shot_dur_max={max(durs):.2f}s"
            )
    except Exception:
        logger.info(f"Handcrafted inputs: shots={num_shots}, duration={float(video_duration):.2f}s")
    
    for i, (start, end) in enumerate(shots_times):
        duration = end - start
        relative_pos = (start + end) / 2 / video_duration if video_duration > 0 else 0.5

        # scene_cut_strength proxy: very short segments often correspond to transitions/cuts.
        # This is a cheap heuristic (no frame-level analysis).
        if duration <= 0:
            cut_strength = 0.0
        elif duration < 1.2:
            cut_strength = 1.0
        elif duration < 3.0:
            cut_strength = 0.75
        else:
            cut_strength = 0.5
        
        # Core temporal features (actually computed)
        shot_features = [
            duration,                  # 1. duration
            relative_pos,              # 2. relative_position
            0.0,                       # 3. motion_mean (placeholder)
            0.0,                       # 4. motion_std (placeholder)
            0.0,                       # 5. motion_peak (placeholder)
            0.0,                       # 6. rms_energy (placeholder)
            0.0,                       # 7. rms_delta (placeholder)
            0.0,                       # 8. spectral_flux (placeholder)
            0.0,                       # 9. pitch_mean (placeholder)
            0.0,                       # 10. pitch_std (placeholder)
            0.0,                       # 11. silence_ratio (placeholder)
            cut_strength,              # 12. scene_cut_strength (duration proxy)
            0.0,                       # 13. color_hist_delta (placeholder)
            0.0                        # 14. is_silent (placeholder)
        ]
        
        features.append(shot_features)
    
    features_tensor = torch.tensor(features, dtype=torch.float32)
    logger.info(f"Extracted handcrafted features: {features_tensor.shape}")
    
    return features_tensor
