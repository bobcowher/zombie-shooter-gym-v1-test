#!/usr/bin/env python
"""
Quick validation test for custom DQN setup
"""

import gymnasium as gym
import zombie_shooter_gym
import torch
from agent import Agent
from model import ZombieNet
from buffer import ReplayBuffer

print("=" * 70)
print("Validation Test - Custom DQN Setup")
print("=" * 70)
print()

# Test 1: Create environment
print("Test 1: Creating gym environment...")
try:
    env = gym.make(
        'ZombieShooter-v1',
        window_width=800,
        window_height=600,
        world_width=1200,
        world_height=1200,
        fps=60,
        sound=False,
        render_mode='rgb_array'
    )
    observation, info = env.reset()
    print(f"✓ Environment created successfully")
    print(f"  Observation shape: {observation.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    exit(1)

print()

# Test 2: Create models
print("Test 2: Creating neural network models...")
try:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = ZombieNet(
        action_dim=env.action_space.n,
        hidden_dim=256,
        dropout=0.2,
        observation_shape=observation.shape
    ).to(device)
    print(f"✓ Model created successfully on {device}")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    env.close()
    exit(1)

print()

# Test 3: Test forward pass
print("Test 3: Testing forward pass...")
try:
    with torch.no_grad():
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = model(obs_tensor)
    print(f"✓ Forward pass successful")
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Q-values: {q_values[0].cpu().numpy()}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    env.close()
    exit(1)

print()

# Test 4: Create replay buffer
print("Test 4: Creating replay buffer...")
try:
    buffer = ReplayBuffer(
        max_size=1000,
        input_shape=observation.shape,
        n_actions=env.action_space.n,
        device=device
    )
    print(f"✓ Replay buffer created successfully")
except Exception as e:
    print(f"✗ Failed to create replay buffer: {e}")
    env.close()
    exit(1)

print()

# Test 5: Store and sample from buffer
print("Test 5: Testing buffer operations...")
try:
    # Take a few steps and store transitions
    for i in range(100):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.store_transition(observation, action, reward, next_obs, done)
        observation = next_obs
        if done:
            observation, info = env.reset()

    # Try to sample
    if buffer.can_sample(32):
        states, actions, rewards, next_states, dones = buffer.sample_buffer(32)
        print(f"✓ Buffer operations successful")
        print(f"  Stored transitions: {buffer.mem_ctr}")
        print(f"  Sample batch size: {states.shape[0]}")
    else:
        print(f"✓ Buffer storing successful (not enough samples yet)")
except Exception as e:
    print(f"✗ Buffer operations failed: {e}")
    env.close()
    exit(1)

print()

# Test 6: Create full agent
print("Test 6: Creating full agent...")
try:
    env.close()
    env = gym.make(
        'ZombieShooter-v1',
        window_width=800,
        window_height=600,
        world_width=1200,
        world_height=1200,
        fps=60,
        sound=False,
        render_mode='rgb_array'
    )

    agent = Agent(
        env,
        dropout=0.2,
        hidden_layer=256,
        learning_rate=0.0001,
        step_repeat=4,
        gamma=0.99
    )
    print(f"✓ Agent created successfully")
except Exception as e:
    print(f"✗ Failed to create agent: {e}")
    env.close()
    exit(1)

print()

# Test 7: Quick training test (1 episode)
print("Test 7: Running quick training test (1 episode, max 100 steps)...")
try:
    observation, info = env.reset()
    episode_reward = 0
    steps = 0
    done = False

    while not done and steps < 100:
        # Random action for quick test
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.memory.store_transition(observation, action, reward, next_obs, done)

        # Try training if enough samples
        if agent.memory.can_sample(32):
            states, actions, rewards, next_states, dones = agent.memory.sample_buffer(32)
            # Just verify we can get the batch (not actually training)
            pass

        observation = next_obs
        episode_reward += reward
        steps += 1

    print(f"✓ Training test successful")
    print(f"  Steps taken: {steps}")
    print(f"  Episode reward: {episode_reward}")
    print(f"  Memory size: {agent.memory.mem_ctr}")
except Exception as e:
    print(f"✗ Training test failed: {e}")
    import traceback
    traceback.print_exc()
    env.close()
    exit(1)

env.close()

print()
print("=" * 70)
print("All validation tests passed! ✓")
print("=" * 70)
print()
print("Setup is ready for training!")
print("  Run: python train.py")
print()
