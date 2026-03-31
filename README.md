# Zombie Shooter Gym - Installation Testing

This directory contains scripts to test the gym-zombie-shooter package installation.

## Setup

1. **Create a clean test environment:**
   ```bash
   conda create -n test-zombie-gym python=3.12
   conda activate test-zombie-gym
   ```

2. **Build the package (from main directory):**
   ```bash
   cd /home/robertcowher/pythonprojects/gym-zombie-shooter-v1
   ./build.sh
   ```

## Testing Options

### Option 1: Test Local Installation (Recommended First)

Test the locally built package before uploading:

```bash
cd /home/robertcowher/pythonprojects/gym-zombie-shooter-v1-test
conda activate test-zombie-gym
./test_from_pypi.sh local
```

### Option 2: Test from TestPyPI

After uploading to TestPyPI:

```bash
conda activate test-zombie-gym
./test_from_pypi.sh testpypi
```

### Option 3: Test from Production PyPI

After publishing to PyPI:

```bash
conda activate test-zombie-gym
./test_from_pypi.sh pypi
```

## Manual Testing

You can also run the test script directly:

```bash
python test_install.py
```

## What Gets Tested

The test script verifies:
1. ✓ Package can be imported
2. ✓ Environment can be created with gym.make()
3. ✓ reset() and step() work correctly
4. ✓ Multiple episodes can run
5. ✓ Observation and action spaces are correct
6. ✓ Rewards and info dict work

## Cleaning Up

To remove the test environment:

```bash
conda deactivate
conda env remove -n test-zombie-gym
```

## Expected Output

When everything works, you should see:

```
============================================================
Zombie Shooter Gym Installation Test
============================================================
Testing import...
✓ Package imported successfully (version 0.1.0)

Testing environment creation...
✓ Environment created successfully
  Observation space: Box(0, 255, (1, 128, 128), uint8)
  Action space: Discrete(7)

Testing reset and step...
✓ Reset works (observation shape: (1, 128, 128))
✓ Step works (reward: 0, terminated: False)

Testing multiple episodes...
  Episode 1: 50 steps, reward: 2.0, health: 5
  Episode 2: 42 steps, reward: -1.0, health: 0
  Episode 3: 50 steps, reward: 1.0, health: 4
✓ Multiple episodes completed successfully

============================================================
Test Summary
============================================================
Passed: 4/4

✓ All tests passed! Package is working correctly.
```
