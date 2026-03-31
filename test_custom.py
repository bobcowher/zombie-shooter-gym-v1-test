#!/usr/bin/env python
"""
Test custom trained DQN models on Zombie Shooter Gym Environment
"""

import sys
import torch
import gymnasium as gym
import zombie_shooter_gym
import numpy as np
from model import ZombieNet


def test_model(
    model1_path='models/dqn1.pt',
    model2_path='models/dqn2.pt',
    n_episodes=5,
    render=False,
    deterministic=True,
    hidden_layer=1024,
    dropout=0.2
):
    """Test custom trained DQN models.

    Args:
        model1_path: Path to first DQN model
        model2_path: Path to second DQN model
        n_episodes: Number of episodes to test
        render: Whether to render the environment
        deterministic: Use deterministic actions (take argmax)
    """

    print("=" * 70)
    print("Testing Custom Double DQN Models")
    print("=" * 70)
    print()

    # Create environment
    print("Creating environment...")
    render_mode = 'human' if render else 'rgb_array'
    env = gym.make(
        'ZombieShooter-v1',
        window_width=1200,
        window_height=800,
        world_width=1800,
        world_height=1200,
        fps=60,
        sound=render,  # Enable sound if rendering
        render_mode=render_mode
    )

    observation, info = env.reset()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Device: {device}")
    print()

    # Load models
    print(f"Loading models...")
    try:
        model_1 = ZombieNet(
            action_dim=env.action_space.n,
            hidden_dim=hidden_layer,
            dropout=dropout,
            observation_shape=observation.shape
        ).to(device)
        model_1.load_the_model(model1_path)

        model_2 = ZombieNet(
            action_dim=env.action_space.n,
            hidden_dim=hidden_layer,
            dropout=dropout,
            observation_shape=observation.shape
        ).to(device)
        model_2.load_the_model(model2_path)

        print("✓ Models loaded successfully")
        print(f"  Model 1: {model1_path}")
        print(f"  Model 2: {model2_path}")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        env.close()
        return

    # Set models to evaluation mode
    model_1.eval()
    model_2.eval()

    print()

    # Test episodes
    episode_rewards = []
    episode_lengths = []

    print("-" * 70)
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Get Q-values from both models and take minimum
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                q_values_1 = model_1(obs_tensor)[0]
                q_values_2 = model_2(obs_tensor)[0]
                q_values = torch.min(q_values_1, q_values_2)

                if deterministic:
                    action = torch.argmax(q_values, dim=-1).item()
                else:
                    # Sample proportional to Q-values
                    probs = torch.softmax(q_values, dim=-1)
                    action = torch.multinomial(probs, 1).item()

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}/{n_episodes}:")
        print(f"  Reward: {episode_reward:6.1f}")
        print(f"  Length: {episode_length:4d} steps")
        print(f"  Health: {info.get('health', 'N/A')}")
        print(f"  Ammo:   {info.get('shotgun_ammo', 'N/A')}")
        print()

    print("-" * 70)
    print()

    # Summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print("Summary Statistics:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Mean length: {mean_length:.1f} steps")
    print(f"  Min reward:  {min(episode_rewards):.1f}")
    print(f"  Max reward:  {max(episode_rewards):.1f}")
    print()

    env.close()

    print("=" * 70)
    print("Testing Complete!")
    print("=" * 70)


def main():
    """Main testing function with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test custom trained DQN models on Zombie Shooter"
    )
    parser.add_argument(
        "--model1",
        type=str,
        default="models/dqn1.pt",
        help="Path to first model file (default: models/dqn1.pt)",
    )
    parser.add_argument(
        "--model2",
        type=str,
        default="models/dqn2.pt",
        help="Path to second model file (default: models/dqn2.pt)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to test (default: 5)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment (watch the agent play)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic",
    )
    parser.add_argument(
        "--hidden-layer",
        type=int,
        default=1024,
        help="Hidden layer size (default: 1024)",
    )

    args = parser.parse_args()

    test_model(
        model1_path=args.model1,
        model2_path=args.model2,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
        hidden_layer=args.hidden_layer,
    )


if __name__ == "__main__":
    main()
