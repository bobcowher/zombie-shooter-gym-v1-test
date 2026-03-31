# Custom Double DQN Training Setup

This repo now contains the original custom Double DQN implementation from zombie-shooter-ai-v1, adapted to work with the Zombie Shooter Gym environment.

## What Was Changed

### Files Copied from Original Project
1. **agent.py** - Double DQN agent implementation
2. **model.py** - ZombieNet CNN architecture
3. **buffer.py** - Replay buffer implementation

### Files Created
1. **train.py** - Custom training script using gym environment
2. **test_custom.py** - Test script for custom trained models
3. **validate_setup.py** - Validation script to ensure setup works

### Modifications Made
1. **agent.py**:
   - Removed unused imports (assets.Zombie, assets.Player)
   - Converted numpy observations to PyTorch tensors
   - Implemented frame skipping manually (gym doesn't have step repeat parameter)

2. **requirements.txt**:
   - Added `pympler>=1.0.1` for memory profiling

3. **Stable Baselines scripts renamed**:
   - `train.py` → `train_sb3.py`
   - `quick_train.py` → `quick_train_sb3.py`

## Training Configuration

The custom training uses the following hyperparameters (matching the original implementation):

- **Episodes**: 10,000
- **Max episode steps**: 2,500 (10,000 / 4 with step repeat)
- **Step repeat**: 4 (frame skipping)
- **Batch size**: 64
- **Learning rate**: 0.0001
- **Epsilon**: 1.0 → 0.1 (decay: 0.995)
- **Gamma**: 0.99
- **Hidden layer**: 1024
- **Dropout**: 0.2
- **Architecture**: Double DQN with two networks and target networks

## Environment Configuration

- **Window size**: 1200x800
- **World size**: 1800x1200
- **FPS**: 60
- **Render mode**: rgb_array
- **Sound**: disabled

## How to Use

### 1. Activate Environment
```bash
conda activate zombie-shooter-test
```

### 2. Validate Setup (Optional)
```bash
python validate_setup.py
```

### 3. Train Custom Models
```bash
python train.py
```

Models will be saved to:
- `models/dqn1.pt`
- `models/dqn2.pt`

TensorBoard logs will be saved to `runs/`

### 4. Monitor Training
```bash
tensorboard --logdir runs/
```

### 5. Test Trained Models
```bash
# Test without rendering
python test_custom.py

# Test with rendering (watch agent play)
python test_custom.py --render --episodes 10

# Custom model paths
python test_custom.py --model1 models/dqn1.pt --model2 models/dqn2.pt
```

## Comparison with Stable Baselines

You can still use the Stable Baselines3 implementation:

```bash
# Train with SB3
python train_sb3.py

# Test SB3 model
python quick_test.py --model trained_models/dqn_final.zip
```

## Key Differences

| Feature | Custom DQN | Stable Baselines3 |
|---------|-----------|-------------------|
| Algorithm | Double DQN (2 networks) | DQN |
| Architecture | Custom CNN (ZombieNet) | Built-in CnnPolicy |
| Frame Skip | 4 frames | 1 frame |
| Replay Buffer | Custom (500K) | Built-in (10K default) |
| Hidden Layer | 1024 | Default (varies) |
| Dropout | 0.2 | None (default) |

## Validation Results

All validation tests passed:
- ✓ Environment creation
- ✓ Model initialization
- ✓ Forward pass
- ✓ Replay buffer operations
- ✓ Agent creation
- ✓ Training loop execution

## Next Steps

1. Run a full training session with `python train.py`
2. Monitor training progress via TensorBoard
3. Compare results with Stable Baselines3 implementation
4. Adjust hyperparameters if needed

## Notes

- The custom implementation uses significantly more memory (15GB+ replay buffer)
- Frame skipping (step_repeat=4) means actual game steps are 4x the training steps
- Models are saved after every episode
- Both model checkpoints (dqn1.pt and dqn2.pt) are needed for testing
