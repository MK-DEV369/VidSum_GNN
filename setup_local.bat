@echo off
REM VidSumGNN Local Setup Script for Windows
REM Automated installation for RTX 3070 with CUDA support

echo ============================================================
echo VidSumGNN Local Setup - RTX 3070 CUDA Installation
echo ============================================================
echo.

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10 or 3.11
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/6] Checking Python version...
python --version

REM Create virtual environment
echo.
echo [2/6] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    echo Virtual environment created successfully!
)

REM Activate virtual environment
echo.
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA 12.1
echo.
echo [5/6] Installing PyTorch with CUDA 12.1 support...
echo This may take a few minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Verify PyTorch installation
echo.
echo Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] PyTorch installation failed!
    echo Try manual installation:
    echo pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pause
    exit /b 1
)

REM Install PyTorch Geometric
echo.
echo [6/6] Installing PyTorch Geometric and other dependencies...
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

REM Install remaining dependencies
echo.
echo Installing transformers and ML libraries...
pip install transformers accelerate

echo.
echo Installing video processing libraries...
pip install opencv-python opencv-contrib-python scenedetect[opencv] ffmpeg-python

echo.
echo Installing audio processing libraries...
pip install librosa soundfile

echo.
echo Installing data science libraries...
pip install numpy pandas matplotlib tqdm scikit-learn

echo.
echo Installing Jupyter (optional)...
pip install jupyterlab ipywidgets notebook

REM Create directory structure
echo.
echo Creating project directories...
if not exist "data\raw\tvsum\video" mkdir data\raw\tvsum\video
if not exist "data\raw\tvsum\ydata" mkdir data\raw\tvsum\ydata
if not exist "data\raw\test_videos" mkdir data\raw\test_videos
if not exist "data\processed\features" mkdir data\processed\features
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "models\pretrained" mkdir models\pretrained
if not exist "results\plots" mkdir results\plots
if not exist "results\logs" mkdir results\logs

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Next Steps:
echo 1. Download TVSum dataset (see LOCAL_SETUP_GUIDE.md)
echo 2. Extract to: data\raw\tvsum\
echo 3. Activate venv: venv\Scripts\activate.bat
echo 4. Run notebook: jupyter lab train.ipynb
echo.
echo To verify installation, run: python verify_setup.py
echo.
echo Virtual environment is now activated.
echo To deactivate later, run: deactivate
echo.

pause
