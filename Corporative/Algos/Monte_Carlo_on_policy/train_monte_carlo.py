#!/usr/bin/env python3
"""
Training script for On-Policy Monte Carlo Control on Cooperative Snake Game
"""

import os
import sys
import argparse
import datetime

# Add current directory to path to enable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from monte_carlo_agent import train_monte_carlo_agent, test_monte_carlo_agent, MonteCarloAgent

def main():
    parser = argparse.ArgumentParser(description='Train Monte Carlo Control agent for Cooperative Snake game')

    # Action arguments
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--test', action='store_true', help='Test the trained agent')
    parser.add_argument('--plot', action='store_true', help='Plot training progress')

    # Game parameters
    parser.add_argument('--width', type=int, default=15, help='Game board width')
    parser.add_argument('--height', type=int, default=15, help='Game board height')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=20000, help='Number of training episodes (default: 100000)')
    parser.add_argument('--save-interval', type=int, default=500, help='Save model every N episodes')
    parser.add_argument('--model-path', type=str, default='models/mc_snake.pkl', help='Path to save/load model')
    parser.add_argument('--load-existing', action='store_true', help='Load existing model if available')
    parser.add_argument('--test-episodes', type=int, default=100, help='Number of test episodes')

    # Algorithm parameters
    parser.add_argument('--gamma', type=float, default=0.90, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Exploration decay rate')

    args = parser.parse_args()

    if not any([args.train, args.test, args.plot]):
        args.train = True # Default action

    if args.train:
        # Create timestamped experiment folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join('models', timestamp)
        os.makedirs(experiment_dir, exist_ok=True)

        print(f"Starting new experiment in: {experiment_dir}")

        # Save hyperparameters
        with open(os.path.join(experiment_dir, 'hyperparameters.txt'), 'w') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")

        print("Training Monte Carlo agent...")

        agent = train_monte_carlo_agent(
            width=args.width,
            height=args.height,
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            model_path=args.model_path,
            load_existing=args.load_existing,
            discount_factor=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            experiment_dir=experiment_dir
        )

        # Plot training progress (final)
        plot_path = os.path.join(experiment_dir, 'training_plot_final.png')
        agent.plot_training_progress(plot_path)

    if args.test:
        print("\nTesting trained agent...")

        # Use the model from the experiment if we just trained, otherwise use the arg
        test_model_path = args.model_path
        if args.train and 'experiment_dir' in locals():
             test_model_path = os.path.join(experiment_dir, 'final_model.pkl')

        test_monte_carlo_agent(
            model_path=test_model_path,
            width=args.width,
            height=args.height,
            num_test_episodes=args.test_episodes
        )

    if args.plot:
        agent = MonteCarloAgent()
        if agent.load_model(args.model_path):
            plot_path = args.model_path.replace('.pkl', '_training.png')
            agent.plot_training_progress(plot_path, show=False)

if __name__ == "__main__":
    main()