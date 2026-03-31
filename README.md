# Zombie Shooter Gym - Validation Repository

This repository validates the `gym-zombie-shooter` environment using the original custom Double DQN implementation from the zombie-shooter-ai-v1 project.

## Purpose

Test whether the gym environment can successfully train an agent using the proven RL implementation from the original game project.

## Prerequisites

The `gym-zombie-shooter` package must be installed separately:

```bash
# Install from PyPI (if published)
pip install gym-zombie-shooter

# OR install from local source
pip install -e /path/to/zombie-shooter-gym-v1
```

## Quick Start

```bash
# 1. Activate your conda environment
conda activate zombie-shooter-test

# 2. Install dependencies
pip install -r requirements.txt

# 3. If using CUDA, install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# 4. Validate setup (optional)
python validate_setup.py

# 5. Run training
python train.py

# 6. Monitor training
tensorboard --logdir runs/

# 7. Test trained models
python test_custom.py --render
```

## What's Included

### Training Files
- **train.py** - Custom Double DQN training script
- **agent.py** - Double DQN agent with dual networks
- **model.py** - ZombieNet CNN architecture
- **buffer.py** - Replay buffer implementation

### Testing Files
- **test_custom.py** - Test custom trained models
- **validate_setup.py** - Validate installation before training
- **test_install.py** - Verify gym environment works

### Configuration
- **requirements.txt** - Python dependencies
- **CUSTOM_TRAINING.md** - Detailed documentation

## Training Details

The custom implementation uses:
- **Algorithm**: Double DQN with two networks + target networks
- **Architecture**: Custom CNN (ZombieNet)
- **Replay Buffer**: 500K transitions (~15GB RAM)
- **Frame Skip**: 4 frames
- **Hidden Layer**: 1024 units
- **Episodes**: 10,000
- **Learning Rate**: 0.0001
- **Batch Size**: 64

See [CUSTOM_TRAINING.md](CUSTOM_TRAINING.md) for complete details.

## Environment Testing

To verify the gym environment installation:

```bash
python test_install.py
```

This tests:
- ✓ Package import
- ✓ Environment creation
- ✓ reset() and step() functionality
- ✓ Multi-episode execution
- ✓ Observation/action spaces
- ✓ Rewards and info dict

## Documentation

- **[CUSTOM_TRAINING.md](CUSTOM_TRAINING.md)** - Complete training guide and configuration details

## Notes

- Models are saved to `models/dqn1.pt` and `models/dqn2.pt` after each episode
- TensorBoard logs go to `runs/` directory
- Both model files are needed for testing
- Training requires ~15GB RAM for replay buffer
