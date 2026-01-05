"""
Optimize dataset directory structure and create useful derived files.
"""

import json
from pathlib import Path
from collections import defaultdict

PROCESSED_DIR = Path("model/data/processed/features")
METADATA_DIR = PROCESSED_DIR / "metadata"
SPLITS_DIR = PROCESSED_DIR / "splits"

def create_domain_splits():
    """Create domain-specific JSON files for easier filtering."""
    
    print("\n" + "=" * 70)
    print("📁 CREATING DOMAIN-SPECIFIC SPLITS")
    print("=" * 70)
    
    with open(METADATA_DIR / "dataset_metadata.json") as f:
        metadata = json.load(f)
    
    # Group by domain
    by_domain = defaultdict(list)
    for video in metadata["videos"]:
        by_domain[video["domain"]].append(video)
    
    # Save domain-specific files
    domain_dir = SPLITS_DIR / "by_domain"
    domain_dir.mkdir(exist_ok=True)
    
    for domain, videos in by_domain.items():
        domain_file = domain_dir / f"{domain}_videos.json"
        with open(domain_file, "w") as f:
            json.dump({
                "domain": domain,
                "num_videos": len(videos),
                "video_ids": [v["video_id"] for v in videos],
                "videos": videos
            }, f, indent=2)
        print(f"✓ Created {domain}_videos.json ({len(videos)} videos)")


def create_summary_statistics():
    """Create comprehensive statistics file."""
    
    print("\n" + "=" * 70)
    print("📊 GENERATING SUMMARY STATISTICS")
    print("=" * 70)
    
    with open(METADATA_DIR / "dataset_metadata.json") as f:
        metadata = json.load(f)
    
    with open(SPLITS_DIR / "train_val_test_split.json") as f:
        splits = json.load(f)
    
    stats = {
        "dataset_summary": {
            "total_videos": metadata["num_videos"],
            "total_shots": sum(v["num_shots"] for v in metadata["videos"]),
            "total_duration_seconds": sum(v["duration"] for v in metadata["videos"]),
            "total_duration_hours": sum(v["duration"] for v in metadata["videos"]) / 3600,
            "avg_shots_per_video": sum(v["num_shots"] for v in metadata["videos"]) / len(metadata["videos"]),
            "avg_duration_per_video": sum(v["duration"] for v in metadata["videos"]) / len(metadata["videos"]),
        },
        "split_distribution": {
            "train": {
                "num_videos": len(splits["train"]),
                "video_ids": splits["train"]
            },
            "val": {
                "num_videos": len(splits["val"]),
                "video_ids": splits["val"]
            },
            "test": {
                "num_videos": len(splits["test"]),
                "video_ids": splits["test"]
            }
        },
        "domain_distribution": {},
        "importance_statistics": {
            "all_videos": {
                "min": min(v["importance_stats"]["min"] for v in metadata["videos"]),
                "max": max(v["importance_stats"]["max"] for v in metadata["videos"]),
                "mean": sum(v["importance_stats"]["mean"] for v in metadata["videos"]) / len(metadata["videos"]),
            }
        }
    }
    
    # Domain stats
    by_domain = defaultdict(lambda: {"videos": [], "total_shots": 0, "total_duration": 0.0})
    for video in metadata["videos"]:
        domain = video["domain"]
        by_domain[domain]["videos"].append(video["video_id"])
        by_domain[domain]["total_shots"] += video["num_shots"]
        by_domain[domain]["total_duration"] += video["duration"]
    
    for domain, data in by_domain.items():
        stats["domain_distribution"][domain] = {
            "num_videos": len(data["videos"]),
            "total_shots": data["total_shots"],
            "total_duration_hours": data["total_duration"] / 3600,
            "avg_shots_per_video": data["total_shots"] / len(data["videos"]),
        }
    
    # Save statistics
    stats_file = METADATA_DIR / "dataset_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Created dataset_statistics.json")
    print(f"\nKey Statistics:")
    print(f"  - Total videos: {stats['dataset_summary']['total_videos']}")
    print(f"  - Total shots: {stats['dataset_summary']['total_shots']}")
    print(f"  - Duration: {stats['dataset_summary']['total_duration_hours']:.2f} hours")
    print(f"  - Avg shots/video: {stats['dataset_summary']['avg_shots_per_video']:.1f}")
    print(f"  - Train/Val/Test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")


def create_quick_index():
    """Create a quick lookup index for video metadata."""
    
    print("\n" + "=" * 70)
    print("🔍 CREATING QUICK LOOKUP INDEX")
    print("=" * 70)
    
    with open(METADATA_DIR / "dataset_metadata.json") as f:
        metadata = json.load(f)
    
    # Create ID -> metadata mapping
    index = {}
    for video in metadata["videos"]:
        index[video["video_id"]] = {
            "duration": video["duration"],
            "domain": video["domain"],
            "split": video["split"],
            "num_shots": video["num_shots"],
            "feature_file": f"features/{video['video_id']}_features.json"
        }
    
    index_file = METADATA_DIR / "video_index.json"
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)
    
    print(f"✓ Created video_index.json ({len(index)} entries)")


def validate_data_integrity():
    """Validate that all referenced files exist and data is consistent."""
    
    print("\n" + "=" * 70)
    print("✅ VALIDATING DATA INTEGRITY")
    print("=" * 70)
    
    with open(PROCESSED_DIR / "complete_dataset.json") as f:
        complete = json.load(f)
    
    with open(METADATA_DIR / "dataset_metadata.json") as f:
        metadata = json.load(f)
    
    issues = []
    
    # Check consistency
    if len(complete) != metadata["num_videos"]:
        issues.append(f"Mismatch: complete_dataset has {len(complete)} but metadata says {metadata['num_videos']}")
    
    # Check feature files exist
    for video in metadata["videos"]:
        feature_file = PROCESSED_DIR / "features" / f"{video['video_id']}_features.json"
        if not feature_file.exists():
            issues.append(f"Missing feature file: {video['video_id']}_features.json")
    
    # Check shot counts match
    for video in complete:
        metadata_video = next((v for v in metadata["videos"] if v["video_id"] == video["video_id"]), None)
        if metadata_video:
            if len(video["shots"]) != metadata_video["num_shots"]:
                issues.append(f"{video['video_id']}: shot count mismatch")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All validation checks passed!")
        print(f"  - {len(complete)} videos verified")
        print(f"  - {sum(len(v['shots']) for v in complete)} shots verified")
        print(f"  - All feature files present")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("⚙️  DATASET OPTIMIZATION & VALIDATION")
    print("=" * 70)
    
    create_domain_splits()
    create_summary_statistics()
    create_quick_index()
    validate_data_integrity()
    
    print("\n" + "=" * 70)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 70)
