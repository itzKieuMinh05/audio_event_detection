#!/bin/bash

# Setup script for Audio Event Detection Project
# This script sets up the complete environment and downloads datasets

echo "=========================================="
echo "Audio Event Detection - Setup Script"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (important for macOS)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision torchaudio

# Install PortAudio for pyaudio support
echo ""
echo "Installing PortAudio for audio input support..."
if command -v brew &> /dev/null; then
    brew install portaudio 2>/dev/null || echo "⚠️  PortAudio installation skipped"
else
    echo "⚠️  Homebrew not found - skipping PortAudio installation"
fi

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt --no-build-isolation 2>&1 | grep -v "SetuptoolsDeprecationWarning"

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed successfully')" || {
    echo "❌ PyTorch verification failed"
    exit 1
}

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/raw/UrbanSound8K
mkdir -p data/raw/ESC-50
#mkdir -p data/raw/FSD50K
mkdir -p data/processed/spectrograms
mkdir -p data/augmented
mkdir -p models/checkpoints
mkdir -p models/saved_models
mkdir -p results/plots
mkdir -p results/metrics
mkdir -p logs

echo ""
echo "=========================================="
echo "Running installation verification test..."
echo "=========================================="
echo ""
python test_installation.py

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download datasets:"
echo "   - UrbanSound8K: https://www.kaggle.com/datasets/chrisfilo/urbansound8k"
echo "   - ESC-50: https://github.com/karolpiczak/ESC-50"
#echo "   - FSD50K: https://zenodo.org/record/4060432"
echo ""
echo "2. Place datasets in data/raw/ directory"
echo ""
echo "3. Run preprocessing:"
echo "   python data/preprocess.py"
echo ""
echo "4. Start training:"
echo "   python scripts/train.py"
echo ""
echo "To rerun the verification test anytime:"
echo "   sh run_test.sh"
echo ""
echo "=========================================="
