#!/usr/bin/env python3
"""
Training script for DQN agent on 2-player Snake game
"""

import os
import sys
import argparse
import datetime

# Add current directory to path to enable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dqn_agent import train_dqn_agent, test_dqn_agent

def main():
    parser = argparse.ArgumentParser(description='Train DQN agent for 2-player Snake game')
    parser.add_argument('--width', type=int, default=15, help='Game board width')
    parser.add_argument('--height', type=int, default=15, help='Game board height')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--save-interval', type=int, default=500, help='Save model every N episodes')
    parser.add_argument('--model-path', type=str, default='models/dqn_snake.pth', help='Path to save/load model (legacy)')
    parser.add_argument('--load-existing', action='store_true', help='Load existing model if available')
    parser.add_argument('--test-only', action='store_true', help='Only test existing model')
    parser.add_argument('--test-episodes', type=int, default=100, help='Number of test episodes')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden dimension size for the MLP')
    parser.add_argument('--train-every', type=int, default=4, help='Optimize model every N steps')
    parser.add_argument('--gamma', type=float, default=0.90, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Exploration decay rate')
    
    args = parser.parse_args()
    
    if args.test_only:
        print("Testing existing DQN model...")
        test_scores, test_lengths = test_dqn_agent(
            model_path=args.model_path,
            width=args.width,
            height=args.height,
            num_test_episodes=args.test_episodes
        )
    else:
        # Create timestamped experiment folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join('models', timestamp)
        os.makedirs(experiment_dir, exist_ok=True)
        
        print(f"Starting new experiment in: {experiment_dir}")
        
        # Save hyperparameters
        with open(os.path.join(experiment_dir, 'hyperparameters.txt'), 'w') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
        
        print("Training DQN agent...")
        
        # Train the agent
        agent = train_dqn_agent(
            width=args.width,
            height=args.height,
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            model_path=args.model_path,
            load_existing=args.load_existing,
            experiment_dir=experiment_dir,
            lr=args.lr,
            hidden_size=args.hidden_size,
            train_every=args.train_every,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay
        )
        
        # Plot training progress (final)
        plot_path = os.path.join(experiment_dir, 'training_plot_final.png')
        agent.plot_training_progress(plot_path)
        
        # Test the trained agent
        print("\nTesting trained agent...")
        test_scores, test_lengths = test_dqn_agent(
            model_path=os.path.join(experiment_dir, 'final_model.pth'),
            width=args.width,
            height=args.height,
            num_test_episodes=50
        )

if __name__ == "__main__":
    main()
