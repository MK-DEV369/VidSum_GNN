# VidSumGNN Local Setup Guide - RTX 3070 🚀

## Prerequisites

- **Hardware**: NVIDIA RTX 3070 GPU
- **OS**: Windows 10/11
- **Python**: 3.10 or 3.11
- **CUDA**: 12.x or 11.8 (will be installed with PyTorch)
- **RAM**: 16GB+ recommended
- **Storage**: ~25GB free space (for dataset + models)

---

## 1. Virtual Environment Setup (HIGHLY RECOMMENDED)

### Why use a Virtual Environment?
- ✅ Isolates project dependencies
- ✅ Prevents conflicts with other Python projects
- ✅ Easy to reproduce environment
- ✅ Clean uninstallation

### Create and Activate venv

```powershell
# Navigate to project directory
cd "E:\5th SEM Data\AI253IA-Artificial Neural Networks and deep learning(ANNDL)\ANN_Project"

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If you get execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try activating again

# Alternative: Command Prompt
# .\venv\Scripts\activate.bat
```

You should see `(venv)` prefix in your terminal when activated.

---

## 2. Install PyTorch with CUDA Support

### Option A: CUDA 12.1 (Recommended for RTX 3070)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option B: CUDA 11.8 (More Stable, Alternative)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```python
python
>>> import torch
>>> print(f"PyTorch Version: {torch.__version__}")
>>> print(f"CUDA Available: {torch.cuda.is_available()}")
>>> print(f"CUDA Version: {torch.version.cuda}")
>>> print(f"GPU: {torch.cuda.get_device_name(0)}")
>>> exit()
```

Expected output:
```
PyTorch Version: 2.1.0+cu121
CUDA Available: True
CUDA Version: 12.1
GPU: NVIDIA GeForce RTX 3070
```

---

## 3. Install PyTorch Geometric

```powershell
# For CUDA 12.1
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# For CUDA 11.8 (alternative)
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

---

## 4. Install Other Dependencies

```powershell
# Transformers and ML libraries
pip install transformers accelerate

# Video processing
pip install opencv-python opencv-contrib-python
pip install scenedetect[opencv]
pip install ffmpeg-python

# Audio processing
pip install librosa soundfile torchaudio

# Utilities
pip install numpy pandas matplotlib tqdm scikit-learn
pip install jupyterlab ipywidgets
```

---

## 5. Download Dataset

### TVSum Dataset (Recommended - Official Benchmark)

**Method 1: Google Drive (Easiest)**

1. Download from: https://drive.google.com/file/d/1bnLq7s6iyWNcGp1l8djKW2JDLqADySqq/view
2. Extract to: `data/raw/tvsum/`

**Method 2: GitHub Repository**

```powershell
# Clone official repository
cd data/raw/
git clone https://github.com/yalesong/tvsum.git tvsum

# Note: Videos may need to be downloaded separately
# Follow instructions in the repository README
```

**Expected Directory Structure:**

```
data/raw/tvsum/
├── ydata/
│   ├── ydata-tvsum50.tsv          # Annotations (importance scores)
│   └── ydata-tvsum50-info.tsv     # Video metadata
└── video/
    ├── 0.mp4                       # 50 videos total
    ├── 1.mp4
    ├── ...
    └── 49.mp4
```

### Dataset Information

- **Videos**: 50 videos (2-10 minutes each)
- **Categories**: 10 types (News, Sports, How-to, Documentary, etc.)
- **Annotations**: 20 human annotators per video
- **Frames**: ~240K frames with importance scores
- **Total Size**: ~10-12GB

### Alternative: SumMe Dataset

```powershell
cd data/raw/
git clone https://github.com/yalesong/vsumm-reinforce.git summe
# Follow repository README for video downloads
```

### Quick Test Without Full Dataset

Place any MP4 videos in `data/raw/test_videos/` for testing:

```powershell
# Create test directory
mkdir -p data/raw/test_videos

# Copy some test videos there
# The training code will auto-generate mock annotations
```

---

## 6. Verify Setup

Create a test script `verify_setup.py`:

```python
import torch
import torch_geometric
from transformers import CLIPModel
import cv2

print("="*60)
print("SYSTEM VERIFICATION")
print("="*60)

# Check PyTorch
print(f"\n✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"✓ GPU Memory: {props.total_memory / 1e9:.2f} GB")

# Check PyTorch Geometric
print(f"\n✓ PyTorch Geometric: {torch_geometric.__version__}")

# Check OpenCV
print(f"✓ OpenCV: {cv2.__version__}")

# Check Transformers (will download model if needed)
print("\n✓ Testing CLIP model download...")
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print("✓ CLIP model ready")
except Exception as e:
    print(f"⚠️ CLIP download issue: {e}")

print("\n" + "="*60)
print("✅ SETUP COMPLETE!")
print("="*60)
```

Run it:
```powershell
python verify_setup.py
```

---

## 7. Launch Jupyter Notebook

```powershell
# Make sure venv is activated
jupyter lab

# Or use VS Code's built-in Jupyter support
# Just open train.ipynb in VS Code
```

---

## 8. Training Commands

### Quick Start Training (in Jupyter)

1. Open `train.ipynb`
2. Run cells sequentially
3. The notebook will:
   - Auto-detect your RTX 3070
   - Load dataset (or use mock data if not available)
   - Configure optimal settings for your GPU
   - Start training with mixed precision (FP16)

### Expected Training Performance

**RTX 3070 Specs:**
- 8GB GDDR6 Memory
- 5888 CUDA Cores
- Tensor Cores: Yes (2nd Gen)

**Training Settings:**
- Batch Size: 4-8 (depends on video length)
- Mixed Precision: FP16 (2x faster than FP32)
- Expected Speed: ~50-100 videos/hour for feature extraction
- Training: ~2-5 minutes per epoch (depends on dataset size)

---

## 9. Troubleshooting

### Issue: CUDA Not Available

```powershell
# Check NVIDIA driver
nvidia-smi

# Should show CUDA Version 12.x or 11.x
# If not, update drivers from: https://www.nvidia.com/Download/index.aspx

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of Memory (OOM)

In the notebook, reduce these values:
```python
config = {
    'batch_size': 2,  # Reduce from 4
    'max_frames_per_video': 100,  # Reduce from 200
}
```

### Issue: Module Not Found

```powershell
# Ensure venv is activated
.\venv\Scripts\Activate.ps1

# Reinstall missing package
pip install <package-name>
```

### Issue: Slow Training on CPU

Verify GPU is detected:
```python
import torch
assert torch.cuda.is_available(), "GPU not detected!"
```

---

## 10. Project Structure

```
ANN_Project/
├── venv/                          # Virtual environment (you create this)
├── data/
│   ├── raw/
│   │   ├── tvsum/                # TVSum dataset (you download this)
│   │   │   ├── video/           # 50 videos
│   │   │   └── ydata/           # Annotations
│   │   └── test_videos/         # Your test videos
│   ├── processed/
│   │   └── features/            # Extracted features (auto-generated)
│   └── outputs/                 # Generated summaries
├── models/
│   ├── checkpoints/             # Training checkpoints (auto-generated)
│   └── pretrained/              # Downloaded models (auto-generated)
├── results/
│   ├── plots/                   # Training plots (auto-generated)
│   └── logs/                    # Training logs (auto-generated)
├── train.ipynb                  # Training notebook (UPDATED)
├── vidsum_gnn/                  # Core library
└── LOCAL_SETUP_GUIDE.md         # This file
```

---

## 11. Next Steps

1. ✅ Setup complete? Run verification script
2. ✅ Dataset downloaded? Check `data/raw/tvsum/video/`
3. ✅ Open `train.ipynb` in VS Code or Jupyter Lab
4. ✅ Run cells sequentially to start training
5. ✅ Monitor GPU usage with `nvidia-smi` in another terminal

---

## 12. Useful Commands

```powershell
# Check GPU usage in real-time
nvidia-smi -l 1

# Activate venv (remember to do this every time)
.\venv\Scripts\Activate.ps1

# Deactivate venv
deactivate

# Install all dependencies at once
pip install -r requirements.txt  # (if you create one)

# Export current environment
pip freeze > requirements.txt

# Clear Python cache
py -m pip cache purge
```

---

## 13. Performance Tips

### For RTX 3070:

1. **Enable Mixed Precision**: Already configured in notebook (FP16)
2. **Optimal Batch Size**: 4-8 for this model
3. **CUDA Optimizations**: 
   ```python
   torch.backends.cudnn.benchmark = True
   torch.backends.cuda.matmul.allow_tf32 = True
   ```
4. **Monitor Memory**: Use `nvidia-smi` or `print_gpu_memory()` function

### Training Speed Expectations:

- Feature Extraction: ~30-60 seconds per video
- Training: ~2-5 minutes per epoch (50 videos)
- Full Training (50 epochs): ~2-4 hours

---

## Support

If you encounter issues:

1. Check GPU with `nvidia-smi`
2. Verify CUDA with `python -c "import torch; print(torch.cuda.is_available())"`
3. Check the troubleshooting section above
4. Ensure virtual environment is activated

---

**Ready to train your video summarization model! 🚀**
