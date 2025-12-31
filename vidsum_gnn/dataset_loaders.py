# Multi-Dataset Loader for TVSum, SumMe, CoSum, OVSum
# Add this to your notebook after the existing VideoDatasetLoader class

import json
import scipy.io as sio
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

class UnifiedVideoDatasetLoader:
    """
    Universal loader for TVSum, SumMe, CoSum, and OVSum datasets.
    Handles heterogeneous annotation formats and normalizes to [0, 1] importance scores.
    """
    
    def __init__(self, base_path='model/data/raw'):
        self.base_path = Path(base_path)
        self.all_videos = []
        
    def load_tvsum(self):
        """Load TVSum dataset (50 videos)"""
        print("\n📥 Loading TVSum Dataset...")
        tvsum_path = self.base_path / 'tvsum'
        video_dir = tvsum_path / 'video'
        anno_file = tvsum_path / 'ydata' / 'ydata-tvsum50.tsv'
        
        if not anno_file.exists():
            print(f"   ❌ TVSum annotations not found: {anno_file}")
            return []
        
        try:
            df = pd.read_csv(anno_file, sep='\t', header=None, 
                           names=['video_id', 'category', 'annotations'])
            print(f"   ✓ Loaded {len(df)} annotation records")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []
        
        dataset = []
        for video_id, group in tqdm(df.groupby('video_id'), desc="TVSum"):
            category = group['category'].iloc[0]
            
            # Find video file
            video_path = self._find_video(video_dir, video_id)
            if video_path is None:
                continue
            
            # Parse scores from all annotators
            all_scores = []
            for ann in group['annotations']:
                scores = [int(x) for x in str(ann).split(',') if x.strip().isdigit()]
                if scores:
                    all_scores.append(scores)
            
            if not all_scores:
                continue
            
            all_scores = np.array(all_scores)
            importance = all_scores.mean(axis=0) / 5.0  # Normalize to [0, 1]
            
            # Get video metadata
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            dataset.append({
                'video_id': video_id,
                'video_path': str(video_path),
                'source': 'tvsum',
                'category': category,
                'importance_scores': importance.tolist(),
                'num_annotators': all_scores.shape[0],
                'fps': fps,
                'frame_count': frame_count,
                'duration': frame_count / fps if fps > 0 else 0
            })
        
        print(f"   ✓ TVSum: {len(dataset)} videos loaded")
        return dataset
    
    def load_summe(self):
        """Load SumMe dataset (25 videos)"""
        print("\n📥 Loading SumMe Dataset...")
        summe_path = self.base_path / 'summe'
        video_dir = summe_path / 'videos'
        gt_dir = summe_path / 'GT'
        
        if not video_dir.exists() or not gt_dir.exists():
            print(f"   ❌ SumMe folders not found")
            return []
        
        try:
            from scipy.io import loadmat
        except ImportError:
            print(f"   ❌ scipy not installed")
            return []
        
        gt_files = sorted(gt_dir.glob('*.mat'))
        dataset = []
        
        for gt in tqdm(gt_files, desc="SumMe"):
            vid = gt.stem
            video_path = self._find_video(video_dir, vid)
            if video_path is None:
                continue
            
            try:
                mat = loadmat(str(gt))
                imp = None
                for key in ['user_score', 'scores', 'gt_scores', 'gtscore']:
                    if key in mat:
                        imp = np.asarray(mat[key]).flatten()
                        break
                
                if imp is None or imp.size == 0:
                    continue
                
                # Normalize to [0, 1]
                imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)
                
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                dataset.append({
                    'video_id': vid,
                    'video_path': str(video_path),
                    'source': 'summe',
                    'category': 'summe',
                    'importance_scores': imp.tolist(),
                    'num_annotators': 1,
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': frame_count / fps if fps > 0 else 0
                })
            except Exception as e:
                print(f"   ⚠️ Error loading {vid}: {e}")
                continue
        
        print(f"   ✓ SumMe: {len(dataset)} videos loaded")
        return dataset
    
    def load_cosum(self):
        """Load CoSum dataset (24 collaborative vlog videos)"""
        print("\n📥 Loading CoSum Dataset...")
        cosum_path = self.base_path / 'cosum'
        video_dir = cosum_path / 'videos'
        anno_dir = cosum_path / 'annotations'
        
        if not video_dir.exists() or not anno_dir.exists():
            print(f"   ❌ CoSum folders not found: {cosum_path}")
            return []
        
        dataset = []
        
        # Try loading frame_labels.json
        labels_file = anno_dir / 'frame_labels.json'
        if labels_file.exists():
            try:
                with open(labels_file, 'r') as f:
                    labels_data = json.load(f)
                
                for vid, data in tqdm(labels_data.items(), desc="CoSum"):
                    video_path = self._find_video(video_dir, vid)
                    if video_path is None:
                        continue
                    
                    # Handle different JSON structures
                    if isinstance(data, list):
                        scores = np.array(data, dtype=float)
                    elif isinstance(data, dict) and 'scores' in data:
                        scores = np.array(data['scores'], dtype=float)
                    else:
                        continue
                    
                    # Normalize to [0, 1]
                    if scores.max() > 1.0:
                        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                    
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    dataset.append({
                        'video_id': vid,
                        'video_path': str(video_path),
                        'source': 'cosum',
                        'category': 'collaborative_vlog',
                        'importance_scores': scores.tolist(),
                        'num_annotators': 3,  # Typical for CoSum
                        'fps': fps,
                        'frame_count': frame_count,
                        'duration': frame_count / fps if fps > 0 else 0
                    })
            except Exception as e:
                print(f"   ⚠️ Error loading CoSum: {e}")
        
        print(f"   ✓ CoSum: {len(dataset)} videos loaded")
        return dataset
    
    def load_ovsum(self):
        """Load OVSum dataset (50 outdoor activity videos)"""
        print("\n📥 Loading OVSum Dataset...")
        ovsum_path = self.base_path / 'ovsum'
        video_dir = ovsum_path / 'videos'
        anno_dir = ovsum_path / 'annotations'
        
        if not video_dir.exists() or not anno_dir.exists():
            print(f"   ❌ OVSum folders not found: {ovsum_path}")
            return []
        
        try:
            from scipy.io import loadmat
        except ImportError:
            print(f"   ❌ scipy not installed")
            return []
        
        dataset = []
        mat_files = sorted(anno_dir.glob('*.mat'))
        
        for mat_file in tqdm(mat_files, desc="OVSum"):
            try:
                vid = mat_file.stem
                video_path = self._find_video(video_dir, vid)
                if video_path is None:
                    continue
                
                mat = loadmat(str(mat_file))
                scores = None
                
                # Try common key names
                for key in ['importance_scores', 'scores', 'gtscore', 'gt_scores']:
                    if key in mat:
                        scores = np.asarray(mat[key]).flatten()
                        break
                
                if scores is None or scores.size == 0:
                    continue
                
                # Normalize to [0, 1]
                if scores.max() > 1.0:
                    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                dataset.append({
                    'video_id': vid,
                    'video_path': str(video_path),
                    'source': 'ovsum',
                    'category': 'outdoor_activity',
                    'importance_scores': scores.tolist(),
                    'num_annotators': 5,  # Typical for OVSum
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': frame_count / fps if fps > 0 else 0
                })
            except Exception as e:
                print(f"   ⚠️ Error loading {mat_file.stem}: {e}")
                continue
        
        print(f"   ✓ OVSum: {len(dataset)} videos loaded")
        return dataset
    
    def load_all(self, include_tvsum=True, include_summe=True, 
                 include_cosum=False, include_ovsum=False):
        """Load multiple datasets and combine"""
        all_data = []
        
        if include_tvsum:
            all_data.extend(self.load_tvsum())
        if include_summe:
            all_data.extend(self.load_summe())
        if include_cosum:
            all_data.extend(self.load_cosum())
        if include_ovsum:
            all_data.extend(self.load_ovsum())
        
        print(f"\n{'='*60}")
        print(f"✅ Total: {len(all_data)} videos loaded")
        print(f"{'='*60}")
        
        # Summary by source
        sources = {}
        for entry in all_data:
            src = entry['source']
            sources[src] = sources.get(src, 0) + 1
        
        for source, count in sorted(sources.items()):
            print(f"   {source.upper():8} → {count:3d} videos")
        
        # Dataset statistics
        total_duration = sum(e['duration'] for e in all_data)
        avg_frames = np.mean([e['frame_count'] for e in all_data])
        print(f"\n   Total duration: {total_duration/3600:.1f} hours")
        print(f"   Avg frames/video: {avg_frames:.0f}")
        
        return all_data
    
    @staticmethod
    def _find_video(video_dir: Path, vid: str):
        """Find video file with any common extension"""
        for ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
            candidate = video_dir / f'{vid}{ext}'
            if candidate.exists():
                return candidate
        
        # Fallback: glob match
        matches = list(video_dir.glob(f'{vid}.*'))
        return matches[0] if matches else None


# =====================================================
# USAGE EXAMPLE
# =====================================================
if __name__ == "__main__":
    loader = UnifiedVideoDatasetLoader(
        base_path=Path('model/data/raw')
    )
    
    # Load your available datasets
    datasets = loader.load_all(
        include_tvsum=True,   # ✓ Download from TVSum official
        include_summe=True,   # ✓ Download from SumMe official
        include_cosum=False,  # Set to True after downloading
        include_ovsum=False   # Set to True after downloading
    )
    
    # Work with combined dataset
    print(f"\n📊 Ready to train on {len(datasets)} videos!")
    
    # Save for training
    import pickle
    with open('model/data/processed/all_datasets.pkl', 'wb') as f:
        pickle.dump(datasets, f)
    print("✓ Saved to model/data/processed/all_datasets.pkl")
