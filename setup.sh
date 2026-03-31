#!/bin/bash
set -e

echo "======================================================================="
echo "Zombie Shooter Gym - Custom Training Setup"
echo "======================================================================="
echo ""

# Install gym-zombie-shooter
echo "Installing gym-zombie-shooter..."
pip install gym-zombie-shooter || {
    echo "WARNING: Could not install from PyPI. Trying local installation..."
    if [ -d "../zombie-shooter-gym-v1" ]; then
        pip install -e ../zombie-shooter-gym-v1
    else
        echo "ERROR: gym-zombie-shooter not available on PyPI and local directory not found"
        exit 1
    fi
}
echo ""

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.9 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
echo ""

# Install remaining dependencies
echo "Installing training dependencies..."
pip install -r requirements.txt
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import zombie_shooter_gym; print(f'Zombie Shooter Gym: {zombie_shooter_gym.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
echo ""

echo "======================================================================="
echo "Setup Complete!"
echo "======================================================================="
echo ""
echo "To validate the setup, run:"
echo "  python validate_setup.py"
echo ""
echo "To start training, run:"
echo "  python train.py"
echo ""
