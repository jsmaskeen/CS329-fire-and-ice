#!/usr/bin/env python3
"""
Training script for Q-Learning agent on 2-player Snake game
"""

import os
import sys
import argparse

# Add current directory to path to enable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Algos.DeepQLearning.DeepQAlgo import train_q_learning_agent, test_q_learning_agent

def main():
    parser = argparse.ArgumentParser(description='Train Q-Learning agent for 2-player Snake game')
    parser.add_argument('--width', type=int, default=15, help='Game board width')
    parser.add_argument('--height', type=int, default=15, help='Game board height')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--save-interval', type=int, default=500, help='Save model every N episodes')
    parser.add_argument('--model-path', type=str, default='models/qlearning_snake.pkl', help='Path to save/load model')
    parser.add_argument('--load-existing', action='store_true', help='Load existing model if available')
    parser.add_argument('--test-only', action='store_true', help='Only test existing model')
    parser.add_argument('--test-episodes', type=int, default=100, help='Number of test episodes')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Exploration decay rate')
    
    args = parser.parse_args()
    
    if args.test_only:
        print("Testing existing Q-Learning model...")
        test_scores, test_lengths = test_q_learning_agent(
            model_path=args.model_path,
            width=args.width,
            height=args.height,
            num_test_episodes=args.test_episodes
        )
    else:
        print("Training Q-Learning agent...")
        
        # Train the agent
        agent = train_q_learning_agent(
            width=args.width,
            height=args.height,
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            model_path=args.model_path,
            load_existing=args.load_existing
        )
        
        # Plot training progress
        base_name, _ = os.path.splitext(args.model_path)
        plot_path = base_name + '_training.png'
        agent.plot_training_progress(plot_path)
        
        # Test the trained agent
        print("\nTesting trained agent...")
        test_scores, test_lengths = test_q_learning_agent(
            model_path=args.model_path,
            width=args.width,
            height=args.height,
            num_test_episodes=50
        )

if __name__ == "__main__":
    main()