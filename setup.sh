#!/bin/bash
set -e

echo "======================================================================="
echo "Zombie Shooter Gym - Custom Training Setup"
echo "======================================================================="
echo ""

# Install PyTorch with CUDA support FIRST
# This must be done before other packages to ensure correct CUDA version
echo "Installing PyTorch with CUDA 12.9 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
echo ""

# Install gym-zombie-shooter from TestPyPI
# Use --extra-index-url to fall back to regular PyPI for dependencies
echo "Installing gym-zombie-shooter from TestPyPI..."
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ gym-zombie-shooter
echo ""

# Install remaining dependencies from requirements.txt
echo "Installing training dependencies..."
pip install -r requirements.txt
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
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
