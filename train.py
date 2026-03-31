#!/usr/bin/env python
"""
Custom DQN Training Script for Zombie Shooter Gym Environment
Uses the original Double DQN implementation from zombie-shooter-ai-v1
"""

import random
import time
import gymnasium as gym
import zombie_shooter_gym
from agent import Agent

# Training hyperparameters
episodes = 10000
max_episode_steps = 10000
step_repeat = 4
max_episode_steps = max_episode_steps / step_repeat

batch_size = 64
learning_rate = 0.0001
epsilon = 1
min_epsilon = 0.1
epsilon_decay = 0.995
gamma = 0.99

hidden_layer = 1024
dropout = 0.2

# Environment parameters
WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800  # Visible game window size
WORLD_WIDTH, WORLD_HEIGHT = 1800, 1200  # The size of the larger game world
FPS = 60

# Create the gym environment
print("=" * 70)
print("Zombie Shooter - Custom Double DQN Training")
print("=" * 70)
print()
print("Creating environment via gym.make()...")

env = gym.make(
    'ZombieShooter-v1',
    window_width=WINDOW_WIDTH,
    window_height=WINDOW_HEIGHT,
    world_width=WORLD_WIDTH,
    world_height=WORLD_HEIGHT,
    fps=FPS,
    sound=False,
    render_mode='rgb_array'
)

print(f"Environment created successfully!")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")
print()

summary_writer_suffix = f'custom_dqn_lr={learning_rate}_hl={hidden_layer}_bs={batch_size}_dropout={dropout}_double_dqn'

print("Initializing agent...")
agent = Agent(
    env,
    dropout=dropout,
    hidden_layer=hidden_layer,
    learning_rate=learning_rate,
    step_repeat=step_repeat,
    gamma=gamma
)

print()
print("Training Configuration:")
print(f"  Episodes: {episodes}")
print(f"  Max episode steps: {int(max_episode_steps)}")
print(f"  Step repeat: {step_repeat}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Hidden layer size: {hidden_layer}")
print(f"  Dropout: {dropout}")
print(f"  Gamma: {gamma}")
print(f"  Initial epsilon: {epsilon}")
print(f"  Min epsilon: {min_epsilon}")
print(f"  Epsilon decay: {epsilon_decay}")
print()
print("=" * 70)
print()

# Training Phase
print("Starting training...")
print("-" * 70)
print()

agent.train(
    episodes=episodes,
    max_episode_steps=max_episode_steps,
    summary_writer_suffix=summary_writer_suffix,
    batch_size=batch_size,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    min_epsilon=min_epsilon
)

print()
print("=" * 70)
print("Training Complete!")
print("=" * 70)
print()
print("Models saved to: models/dqn1.pt and models/dqn2.pt")
print(f"Tensorboard logs: runs/*{summary_writer_suffix}*")
print()
print("To view training progress:")
print("  tensorboard --logdir runs/")
print()
