#!/usr/bin/env python
"""Test script to verify gym-zombie-shooter installation."""

import sys

def test_import():
    """Test that the package can be imported."""
    print("Testing import...")
    try:
        import zombie_shooter_gym
        print(f"✓ Package imported successfully (version {zombie_shooter_gym.__version__})")
        return True
    except ImportError as e:
        print(f"✗ Failed to import package: {e}")
        return False


def test_environment_creation():
    """Test that the environment can be created."""
    print("\nTesting environment creation...")
    try:
        import zombie_shooter_gym
        import gymnasium as gym

        env = gym.make('ZombieShooter-v1', render_mode='rgb_array')
        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False


def test_reset_and_step():
    """Test that reset and step work."""
    print("\nTesting reset and step...")
    try:
        import zombie_shooter_gym
        import gymnasium as gym
        import numpy as np

        env = gym.make('ZombieShooter-v1', render_mode='rgb_array')

        # Test reset
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert obs.shape == (1, 128, 128), f"Expected shape (1, 128, 128), got {obs.shape}"
        print(f"✓ Reset works (observation shape: {obs.shape})")

        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        print(f"✓ Step works (reward: {reward}, terminated: {terminated})")

        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed reset/step test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_episodes():
    """Test running multiple episodes."""
    print("\nTesting multiple episodes...")
    try:
        import zombie_shooter_gym
        import gymnasium as gym

        env = gym.make('ZombieShooter-v1', render_mode='rgb_array')

        for episode in range(3):
            obs, info = env.reset()
            total_reward = 0
            steps = 0

            for step in range(50):  # Run 50 steps per episode
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            print(f"  Episode {episode + 1}: {steps} steps, reward: {total_reward:.1f}, health: {info['health']}")

        env.close()
        print("✓ Multiple episodes completed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed multiple episodes test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Zombie Shooter Gym Installation Test")
    print("=" * 60)

    tests = [
        test_import,
        test_environment_creation,
        test_reset_and_step,
        test_multiple_episodes,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! Package is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
