"""
YouTube Dataset Builder - Comprehensive Test Suite
Tests all pipeline components individually
"""

import json
import sys
import os
from pathlib import Path
import tempfile
import numpy as np
from dataclasses import dataclass

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import from youtube_dataset
try:
    from youtube_dataset import (
        ShotFeatures, Shot, VideoDataset,
        validate_dataset, save_dataset_structure, test_dataset,
        normalize_features, compute_importance, smooth_importance_scores,
        assign_ranks, TOP_5_PLAYLISTS
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure youtube_dataset.py is in the same directory")
    sys.exit(1)


def test_dataclasses():
    """Test ShotFeatures, Shot, VideoDataset dataclasses"""
    print("\n" + "="*70)
    print("TEST 1: Data Structure Validation")
    print("="*70)
    
    try:
        # Test ShotFeatures
        features = ShotFeatures(
            motion=0.5,
            speech=0.7,
            scene_change=1.0,
            audio_energy=0.6,
            object_count=0.3
        )
        assert hasattr(features, 'to_dict'), "ShotFeatures missing to_dict()"
        assert callable(features.to_dict), "to_dict() not callable"
        
        # Test Shot
        shot = Shot(
            start=0.0,
            end=5.0,
            features=features,
            importance=0.65,
            rank=1
        )
        assert shot.start < shot.end, "Shot: start >= end"
        
        # Test VideoDataset
        video = VideoDataset(
            video_id="test_123",
            duration=100.0,
            domain="lecture",
            shots=[shot]
        )
        assert hasattr(video, 'to_dict'), "VideoDataset missing to_dict()"
        assert len(video.shots) == 1, "Shot count mismatch"
        
        print("✓ All dataclasses validated")
        return True
    except Exception as e:
        print(f"✗ Dataclass test failed: {e}")
        return False


def test_feature_normalization():
    """Test normalize_features function"""
    print("\n" + "="*70)
    print("TEST 2: Feature Normalization")
    print("="*70)
    
    try:
        # Create ShotFeatures objects
        features = [
            ShotFeatures(motion=0.1, speech=0.2, scene_change=0.0, audio_energy=0.3, object_count=0.5),
            ShotFeatures(motion=0.5, speech=0.6, scene_change=1.0, audio_energy=0.7, object_count=0.5),
            ShotFeatures(motion=0.9, speech=0.8, scene_change=0.0, audio_energy=0.4, object_count=0.5),
        ]
        
        normalized = normalize_features(features)
        
        # Check normalization - should return list of dicts
        assert isinstance(normalized, list), "Should return list"
        assert isinstance(normalized[0], dict), "Should return list of dicts"
        
        for feature_name in ['motion', 'speech', 'scene_change', 'audio_energy', 'object_count']:
            values = [n[feature_name] for n in normalized]
            min_val = min(values)
            max_val = max(values)
            assert min_val >= -0.01 and max_val <= 1.01, \
                f"{feature_name}: values not in [0,1]"
        
        print("✓ Feature normalization passed")
        motion_vals = [n['motion'] for n in normalized]
        speech_vals = [n['speech'] for n in normalized]
        print(f"  - Motion range: [{min(motion_vals):.3f}, {max(motion_vals):.3f}]")
        print(f"  - Speech range: [{min(speech_vals):.3f}, {max(speech_vals):.3f}]")
        return True
    except Exception as e:
        print(f"✗ Normalization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_importance_scoring():
    """Test compute_importance function"""
    print("\n" + "="*70)
    print("TEST 3: Importance Scoring")
    print("="*70)
    
    try:
        features = ShotFeatures(
            motion=0.8,
            speech=0.3,
            scene_change=1.0,
            audio_energy=0.5,
            object_count=0.4
        )
        
        # Test different domains
        domains = ['lecture', 'interview', 'sports', 'documentary', 'default']
        
        for domain in domains:
            importance = compute_importance(features, domain=domain)
            assert 0 <= importance <= 1, f"Importance out of range for {domain}"
            print(f"✓ {domain:12} → importance: {importance:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Importance scoring test failed: {e}")
        return False


def test_temporal_smoothing():
    """Test smooth_importance_scores function"""
    print("\n" + "="*70)
    print("TEST 4: Temporal Smoothing")
    print("="*70)
    
    try:
        # Create spiky importance scores
        scores = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        
        smoothed = smooth_importance_scores(scores, sigma=1.0)
        
        # Check that smoothing reduces variance
        orig_variance = np.var(scores)
        smooth_variance = np.var(smoothed)
        
        assert smooth_variance < orig_variance, "Smoothing didn't reduce variance"
        print(f"✓ Temporal smoothing passed")
        print(f"  - Original variance: {orig_variance:.3f}")
        print(f"  - Smoothed variance: {smooth_variance:.3f}")
        print(f"  - Original: {scores}")
        print(f"  - Smoothed: {np.round(smoothed, 3)}")
        
        return True
    except Exception as e:
        print(f"✗ Temporal smoothing test failed: {e}")
        return False


def test_rank_assignment():
    """Test assign_ranks function"""
    print("\n" + "="*70)
    print("TEST 5: Rank Assignment")
    print("="*70)
    
    try:
        # Create shots with different importance
        features = ShotFeatures(0.5, 0.5, 0.5, 0.5, 0.5)
        shots = [
            Shot(0.0, 5.0, features, importance=0.9, rank=0),
            Shot(5.0, 10.0, features, importance=0.3, rank=0),
            Shot(10.0, 15.0, features, importance=0.7, rank=0),
        ]
        
        ranked_shots = assign_ranks(shots)
        
        # Check ranks are assigned correctly (sorted by importance descending)
        importance_values = [s.importance for s in ranked_shots]
        ranks = [s.rank for s in ranked_shots]
        
        # Find which shot has rank 1 (should be the one with importance 0.9)
        rank1_shot = next(s for s in ranked_shots if s.rank == 1)
        assert rank1_shot.importance == 0.9, f"Rank 1 should have importance 0.9, got {rank1_shot.importance}"
        
        rank2_shot = next(s for s in ranked_shots if s.rank == 2)
        assert rank2_shot.importance == 0.7, f"Rank 2 should have importance 0.7, got {rank2_shot.importance}"
        
        rank3_shot = next(s for s in ranked_shots if s.rank == 3)
        assert rank3_shot.importance == 0.3, f"Rank 3 should have importance 0.3, got {rank3_shot.importance}"
        
        print("✓ Rank assignment passed")
        for i, shot in enumerate(ranked_shots):
            print(f"  Shot {i}: importance={shot.importance:.3f}, rank={shot.rank}")
        
        return True
    except Exception as e:
        print(f"✗ Rank assignment test failed: {e}")
        return False


def test_dataset_validation():
    """Test validate_dataset and test_dataset functions"""
    print("\n" + "="*70)
    print("TEST 6: Dataset Validation")
    print("="*70)
    
    try:
        # Create mock dataset
        features = ShotFeatures(0.5, 0.5, 0.5, 0.5, 0.5)
        shots = [
            Shot(i*5.0, (i+1)*5.0, features, importance=np.random.rand(), rank=i)
            for i in range(5)
        ]
        
        dataset = [
            VideoDataset(
                video_id=f"video_{j}",
                duration=25.0,
                domain="lecture",
                shots=shots
            )
            for j in range(3)
        ]
        
        # Validate
        stats = validate_dataset(dataset)
        assert stats['valid'], "Dataset validation failed"
        
        print("✓ Dataset validation passed")
        print(f"  - Videos: {stats['num_videos']}")
        print(f"  - Total shots: {stats['total_shots']}")
        print(f"  - Avg shots/video: {stats['avg_shots_per_video']:.1f}")
        print(f"  - Importance range: [{stats['importance_stats']['min']:.3f}, {stats['importance_stats']['max']:.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Dataset validation test failed: {e}")
        return False


def test_directory_structure():
    """Test directory structure creation"""
    print("\n" + "="*70)
    print("TEST 7: Directory Structure")
    print("="*70)
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock dataset
            features = ShotFeatures(0.5, 0.5, 0.5, 0.5, 0.5)
            shot = Shot(0.0, 5.0, features, importance=0.7, rank=1)
            dataset = [
                VideoDataset("test_vid", 5.0, "lecture", [shot])
            ]
            
            # Save structure
            paths = save_dataset_structure(dataset, tmpdir)
            
            # Verify structure
            assert Path(paths['dataset']).exists(), "Dataset file not created"
            assert Path(paths['metadata']).exists(), "Metadata file not created"
            assert Path(paths['features_dir']).exists(), "Features dir not created"
            assert Path(paths['splits']).exists(), "Splits file not created"
            
            print("✓ Directory structure creation passed")
            for name, path in paths.items():
                exists = "✓" if Path(path).exists() else "✗"
                print(f"  {exists} {name}: {path}")
            
            return True
    except Exception as e:
        print(f"✗ Directory structure test failed: {e}")
        return False


def test_playlist_config():
    """Test TOP_5_PLAYLISTS configuration"""
    print("\n" + "="*70)
    print("TEST 8: Playlist Configuration")
    print("="*70)
    
    try:
        assert len(TOP_5_PLAYLISTS) == 5, f"Expected 5 playlists, got {len(TOP_5_PLAYLISTS)}"
        
        for name, config in TOP_5_PLAYLISTS.items():
            assert 'url' in config, f"{name}: missing 'url'"
            assert 'domain' in config, f"{name}: missing 'domain'"
            assert 'description' in config, f"{name}: missing 'description'"
            
            valid_domains = {'lecture', 'interview', 'sports', 'documentary', 'default'}
            assert config['domain'] in valid_domains, \
                f"{name}: invalid domain '{config['domain']}'"
            
            assert 'youtube.com/playlist' in config['url'], \
                f"{name}: invalid playlist URL"
            
            print(f"✓ {name:20} - {config['domain']:12} - {config['description']}")
        
        return True
    except Exception as e:
        print(f"✗ Playlist configuration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 YOUTUBE DATASET BUILDER - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Data Structures", test_dataclasses),
        ("Feature Normalization", test_feature_normalization),
        ("Importance Scoring", test_importance_scoring),
        ("Temporal Smoothing", test_temporal_smoothing),
        ("Rank Assignment", test_rank_assignment),
        ("Dataset Validation", test_dataset_validation),
        ("Directory Structure", test_directory_structure),
        ("Playlist Configuration", test_playlist_config),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n✅ All tests passed! Pipeline is ready to use.")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} test(s) failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
