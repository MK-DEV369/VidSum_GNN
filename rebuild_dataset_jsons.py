"""
Rebuild complete_dataset.json and dataset_metadata.json from individual feature files.
"""

import json
from pathlib import Path
import subprocess

FEATURES_DIR = Path("model/data/processed/features/features")
HIGH_IMPORTANCE_DIR = Path("model/data/raw/youtube/high_importance")
OUTPUT_DIR = Path("model/data/processed/features")
METADATA_DIR = OUTPUT_DIR / "metadata"
SPLITS_DIR = OUTPUT_DIR / "splits"

# Domain mapping from video source
DOMAIN_MAP = {
    # Sports (ESPN)
    "Qd8v8hZsCJk": "sports",
    "uPYKrIgXBuw": "sports",
    
    # Documentary (Kurzgesagt)
    "21eFwbb48sE": "documentary",
    "2XkV6IpV2Y0": "documentary",
    "4_aOIA-vyBo": "documentary",
    "F3QpgXBtDeo": "documentary",
    "hOfRN0KihOU": "documentary",
    "KsF_hdjWJjo": "documentary",
    "Uti2niW2BRA": "documentary",
    "UuGrBhK2c7U": "documentary",
    "wNDGgL73ihY": "documentary",
    "xRF7WIZV4lA": "documentary",
    
    # Lecture (TED Talks)
    "11RKiZ1S0e4": "lecture",
    "6DNqKig7Xis": "lecture",
    "7XFLTDQ4JMk": "lecture",
    "8KkKuTCFvzI": "lecture",
    "aaXBYcfVYZM": "lecture",
    "iCvmsMzlF7o": "lecture",
    "Lp7E973zozc": "lecture",
    "TFbv757kup4": "lecture",
    "W_x9cbrdgnw": "lecture",
    "xmndCPvXiik": "lecture",
    
    # High Importance (curated for quality/importance)
    "8ZfBKk7YneU": "high_importance",
    "JfVOs4VSpmA": "high_importance",
    "JIRqdeNl2cU": "high_importance",
    "oq02OFHhTSE": "high_importance",
    "s7WPMv2IgFk": "high_importance",
    "sNPnbI1arSE": "high_importance",
    "VQRLujxTm3c": "high_importance",
    "XbGs_qK2PQA": "high_importance",
    "yzr2rXRGJz8": "high_importance",
}


def _assign_split(idx: int) -> str:
    """Deterministically assign train/val/test split."""
    mod = idx % 10
    if mod < 6:
        return "train"
    if mod < 8:
        return "val"
    return "test"


def infer_domain_from_features(video_id: str, features_data: dict) -> str:
    """Infer domain from feature patterns if not in DOMAIN_MAP."""
    if video_id in DOMAIN_MAP:
        return DOMAIN_MAP[video_id]
    
    # Analyze first few shots to infer domain
    if not features_data.get("shots"):
        return "default"
    
    shots = features_data["shots"][:10]  # Sample first 10 shots
    avg_motion = sum(s["features"]["motion"] for s in shots) / len(shots)
    avg_speech = sum(s["features"]["speech"] for s in shots) / len(shots)
    
    # High motion, low speech = sports/gaming
    if avg_motion > 0.4 and avg_speech < 0.3:
        return "sports"
    # Low motion, high speech = lecture/interview
    elif avg_motion < 0.1 and avg_speech > 0.4:
        return "lecture"
    # Balanced = documentary
    elif 0.2 <= avg_motion <= 0.4 and 0.3 <= avg_speech <= 0.5:
        return "documentary"
    
    return "default"


def get_duration_from_info_json(video_id: str) -> float:
    """Extract duration from .info.json file for high importance videos."""
    info_file = HIGH_IMPORTANCE_DIR / f"{video_id}.info.json"
    if info_file.exists():
        try:
            with open(info_file) as f:
                info_data = json.load(f)
                return float(info_data.get("duration", 0.0))
        except Exception as e:
            print(f"  [WARN] Could not read duration from {info_file}: {e}")
            return 0.0
    return 0.0


def rebuild_jsons():
    """Rebuild complete_dataset.json and dataset_metadata.json from feature files and high importance videos."""
    
    print("=" * 70)
    print("🔧 REBUILDING DATASET JSON FILES (including high importance videos)")
    print("=" * 70)
    
    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing splits or create new ones
    splits_file = SPLITS_DIR / "train_val_test_split.json"
    if splits_file.exists():
        with open(splits_file) as f:
            splits = json.load(f)
        existing_videos = set(splits["train"] + splits["val"] + splits["test"])
        print(f"✓ Loaded existing splits ({len(existing_videos)} videos)")
    else:
        splits = {"train": [], "val": [], "test": []}
        existing_videos = set()
        print("Creating new splits file")
    
    # Process all feature files (including recursive search in subdirectories)
    feature_files = sorted(FEATURES_DIR.glob("**/*_features.json"))
    print(f"\nFound {len(feature_files)} feature files in {FEATURES_DIR}")
    
    # Get high importance videos
    high_importance_videos = set([f.stem for f in HIGH_IMPORTANCE_DIR.glob("*.info.json")])
    print(f"Found {len(high_importance_videos)} high importance videos in {HIGH_IMPORTANCE_DIR}")
    
    complete_dataset = []
    metadata_videos = []
    new_videos = 0
    updated_videos = 0
    
    # Process regular feature files (remove duplicates by video_id)
    processed_videos = set()
    
    for idx, feature_file in enumerate(feature_files):
        video_id = feature_file.stem.replace("_features", "")
        
        # Skip if already processed
        if video_id in processed_videos:
            continue
        processed_videos.add(video_id)
        
        try:
            with open(feature_file) as f:
                features_data = json.load(f)
            
            # Determine domain
            domain = infer_domain_from_features(video_id, features_data)
            
            # Track if this is a new video
            is_new = video_id not in existing_videos
            
            # Determine split (use existing or assign new)
            if video_id in existing_videos:
                if video_id in splits["train"]:
                    split = "train"
                elif video_id in splits["val"]:
                    split = "val"
                else:
                    split = "test"
                updated_videos += 1
            else:
                split = _assign_split(len(splits["train"]) + len(splits["val"]) + len(splits["test"]))
                splits[split].append(video_id)
                new_videos += 1
            
            # Build complete dataset entry
            video_entry = {
                "video_id": video_id,
                "duration": features_data["duration"],
                "domain": domain,
                "shots": features_data["shots"]
            }
            complete_dataset.append(video_entry)
            
            # Build metadata entry
            shots = features_data["shots"]
            importance_scores = [s["importance"] for s in shots]
            
            metadata_entry = {
                "video_id": video_id,
                "duration": features_data["duration"],
                "domain": domain,
                "split": split,
                "num_shots": len(shots),
                "importance_stats": {
                    "min": float(min(importance_scores)) if importance_scores else 0.0,
                    "max": float(max(importance_scores)) if importance_scores else 0.0,
                    "mean": float(sum(importance_scores) / len(importance_scores)) if importance_scores else 0.0,
                }
            }
            metadata_videos.append(metadata_entry)
            
            status = "NEW" if is_new else "UPD"
            print(f"  [{status}] {video_id}: {domain:15s} {split:5s} {len(shots):3d} shots")
            
        except Exception as e:
            print(f"  [ERR] {video_id}: {e}")
    
    # Process high importance videos
    print(f"\nProcessing {len(high_importance_videos)} high importance videos...")
    for video_id in sorted(high_importance_videos):
        try:
            # Skip if already processed
            if video_id in processed_videos:
                continue
            processed_videos.add(video_id)
            
            domain = DOMAIN_MAP.get(video_id, "high_importance")
            duration = get_duration_from_info_json(video_id)
            
            # Look for feature file for this video (search recursively)
            feature_files_for_video = list(FEATURES_DIR.glob(f"**/{video_id}_features.json"))
            shots = []
            importance_stats = {"min": 0.0, "max": 0.0, "mean": 0.0}
            num_shots = 0
            
            if feature_files_for_video:
                feature_file = feature_files_for_video[0]  # Take first match
                try:
                    with open(feature_file) as f:
                        features_data = json.load(f)
                        shots = features_data.get("shots", [])
                        num_shots = len(shots)
                        if shots:
                            importance_scores = [s["importance"] for s in shots]
                            importance_stats = {
                                "min": float(min(importance_scores)),
                                "max": float(max(importance_scores)),
                                "mean": float(sum(importance_scores) / len(importance_scores)),
                            }
                except Exception as e:
                    print(f"  [WARN] Could not load features for {video_id}: {e}")
            
            # Track if this is a new video
            is_new = video_id not in existing_videos
            
            # Determine split (use existing or assign new)
            if video_id in existing_videos:
                if video_id in splits["train"]:
                    split = "train"
                elif video_id in splits["val"]:
                    split = "val"
                else:
                    split = "test"
                updated_videos += 1
            else:
                split = _assign_split(len(splits["train"]) + len(splits["val"]) + len(splits["test"]))
                splits[split].append(video_id)
                new_videos += 1
            
            # Build complete dataset entry
            video_entry = {
                "video_id": video_id,
                "duration": duration,
                "domain": domain,
                "shots": shots
            }
            complete_dataset.append(video_entry)
            
            # Build metadata entry
            metadata_entry = {
                "video_id": video_id,
                "duration": duration,
                "domain": domain,
                "split": split,
                "num_shots": num_shots,
                "importance_stats": importance_stats
            }
            metadata_videos.append(metadata_entry)
            
            status = "NEW" if is_new else "UPD"
            shot_info = f"{num_shots} shots" if num_shots > 0 else "(no feature file)"
            print(f"  [{status}] {video_id}: {domain:15s} {split:5s} {shot_info}")
            
        except Exception as e:
            print(f"  [ERR] {video_id}: {e}")
    
    # Sort by split and video_id
    metadata_videos.sort(key=lambda x: (x["split"], x["video_id"]))
    complete_dataset.sort(key=lambda x: x["video_id"])
    
    # Save complete_dataset.json
    complete_path = OUTPUT_DIR / "complete_dataset.json"
    with open(complete_path, "w") as f:
        json.dump(complete_dataset, f, indent=2)
    print(f"\n✓ Saved complete_dataset.json ({len(complete_dataset)} videos)")
    
    # Save dataset_metadata.json
    metadata = {
        "num_videos": len(metadata_videos),
        "videos": metadata_videos
    }
    metadata_path = METADATA_DIR / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved dataset_metadata.json ({len(metadata_videos)} videos)")
    
    # Update splits file
    with open(splits_file, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"✓ Updated train_val_test_split.json")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("📊 DATASET STATISTICS")
    print("=" * 70)
    print(f"Total videos: {len(complete_dataset)} ({new_videos} new, {updated_videos} updated)")
    print(f"Total shots: {sum(len(v['shots']) for v in complete_dataset)}")
    print(f"Total duration: {sum(v['duration'] for v in complete_dataset) / 3600:.2f} hours")
    print(f"\nSplit distribution:")
    print(f"  - Train: {len(splits['train'])} videos")
    print(f"  - Val:   {len(splits['val'])} videos")
    print(f"  - Test:  {len(splits['test'])} videos")
    print(f"\nDomain distribution:")
    domain_counts = {}
    for v in metadata_videos:
        domain_counts[v["domain"]] = domain_counts.get(v["domain"], 0) + 1
    for domain, count in sorted(domain_counts.items()):
        print(f"  - {domain.capitalize():20s}: {count:3d} videos")
    print("=" * 70)


if __name__ == "__main__":
    rebuild_jsons()
