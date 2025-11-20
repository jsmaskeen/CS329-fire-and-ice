#!/usr/bin/env python3
"""
Training script for On-Policy Monte Carlo Control on Cooperative Snake Game
"""

import os
import sys
import argparse

# Add current directory to path to enable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MonteCarloOnpolicyalgo import train_mc_control, test_mc_control, MonteCarloAgent

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
    parser.add_argument('--episodes', type=int, default=100000, help='Number of training episodes (default: 100000)')
    parser.add_argument('--save-interval', type=int, default=10000, help='Save model every N episodes')
    parser.add_argument('--model-path', type=str, default='models/mc_snake.pkl', help='Path to save/load model')
    parser.add_argument('--load-existing', action='store_true', help='Load existing model if available')
    parser.add_argument('--test-episodes', type=int, default=100, help='Number of test episodes')

    # Algorithm parameters
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')

    args = parser.parse_args()

    if not any([args.train, args.test, args.plot]):
        args.train = True # Default action

    if args.train:
        print("Training Monte Carlo agent...")

        agent = train_mc_control(
            width=args.width,
            height=args.height,
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            model_path=args.model_path,
            load_existing=args.load_existing,
            discount_factor=args.gamma
        )

    if args.test:
        print("\nTesting trained agent...")
        test_mc_control(
            model_path=args.model_path,
            width=args.width,
            height=args.height,
            num_test_episodes=args.test_episodes
        )

    if args.plot:
        agent = MonteCarloAgent()
        if agent.load_model(args.model_path):
            plot_path = args.model_path.replace('.pkl', '_training.png')
            agent.plot_training_progress(plot_path)

if __name__ == "__main__":
    main()