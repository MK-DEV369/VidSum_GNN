#!/bin/bash
# YouTube Dataset Setup Script
# Installs all dependencies and configures environment

echo "=========================================="
echo "YouTube Dataset Builder Setup"
echo "=========================================="

# Install Python packages
echo ""
echo "Installing Python dependencies..."
pip install yt-dlp librosa opencv-python scenedetect scipy numpy --upgrade

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p model/data/videos/{TED-Talks,Kurzgesagt,CNN-Breaking-News,ESPN-Highlights,BBC-Learning}
mkdir -p model/data/metadata
mkdir -p model/data/features
mkdir -p model/data/splits
mkdir -p model/data/processed

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download videos using provided yt-dlp commands (see YOUTUBE_DATASET_README.md)"
echo "2. Run: python youtube_dataset.py"
echo "3. Check output in model/data/"
echo ""
