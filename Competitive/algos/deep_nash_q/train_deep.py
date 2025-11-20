#!/usr/bin/env python3
"""
Training script for Deep Nash Q-Learning on Competitive Snake Game
"""

import sys
import os
import argparse

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deep_nash_qlearning import train_deep_nash_q_learning, DeepNashQLearningAgent
from game import CompetitiveSnakeGame
import numpy as np


def test_agents(model_dir='models', width=15, height=15, num_episodes=100):
    """
    Test trained Deep Nash Q-Learning agents
    
    Args:
        model_dir: Directory containing trained models
        width: Game board width
        height: Game board height
        num_episodes: Number of test episodes
    """
    print(f"\nTesting trained Deep Nash Q-Learning agents for {num_episodes} episodes...")
    
    # Load agents
    agent1 = DeepNashQLearningAgent(agent_id=1)
    agent2 = DeepNashQLearningAgent(agent_id=2)
    
    if not agent1.load(f"{model_dir}/agent1_deep_nash.pth"):
        print("Error: Agent 1 model not found!")
        return
    if not agent2.load(f"{model_dir}/agent2_deep_nash.pth"):
        print("Error: Agent 2 model not found!")
        return
    
    # Set to pure exploitation
    agent1.epsilon = 0.0
    agent2.epsilon = 0.0
    
    env = CompetitiveSnakeGame(width, height)
    
    test_results = {
        'agent1_wins': 0,
        'agent2_wins': 0,
        'draws': 0,
        'agent1_scores': [],
        'agent2_scores': [],
        'episode_lengths': []
    }
    
    for episode in range(num_episodes):
        state1, state2 = env.reset()
        steps = 0
        
        while True:
            action1 = agent1.get_action(state1, training=False)
            action2 = agent2.get_action(state2, training=False)
            
            reward1, reward2, (next_state1, next_state2), done, info = \
                env.step(action1, action2)
            
            state1 = next_state1
            state2 = next_state2
            steps += 1
            
            if done:
                break
        
        # Record results
        test_results['agent1_scores'].append(info['snake1_score'])
        test_results['agent2_scores'].append(info['snake2_score'])
        test_results['episode_lengths'].append(steps)
        
        if info['snake1_score'] > info['snake2_score']:
            test_results['agent1_wins'] += 1
        elif info['snake2_score'] > info['snake1_score']:
            test_results['agent2_wins'] += 1
        else:
            test_results['draws'] += 1
        
        if episode % 10 == 0:
            print(f"Test Episode {episode:3d} | "
                  f"Agent1: {info['snake1_score']:2d} | "
                  f"Agent2: {info['snake2_score']:2d} | "
                  f"Steps: {steps:3d}")
    
    # Print summary
    print("\n" + "="*60)
    print("DEEP NASH Q-LEARNING TEST RESULTS")
    print("="*60)
    print(f"Agent 1 Wins: {test_results['agent1_wins']} ({test_results['agent1_wins']/num_episodes*100:.1f}%)")
    print(f"Agent 2 Wins: {test_results['agent2_wins']} ({test_results['agent2_wins']/num_episodes*100:.1f}%)")
    print(f"Draws: {test_results['draws']} ({test_results['draws']/num_episodes*100:.1f}%)")
    print(f"\nAgent 1 Avg Score: {np.mean(test_results['agent1_scores']):.2f} ± {np.std(test_results['agent1_scores']):.2f}")
    print(f"Agent 2 Avg Score: {np.mean(test_results['agent2_scores']):.2f} ± {np.std(test_results['agent2_scores']):.2f}")
    print(f"\nAverage Episode Length: {np.mean(test_results['episode_lengths']):.1f} ± {np.std(test_results['episode_lengths']):.1f}")
    print("="*60)
    
    return test_results


def main():
    parser = argparse.ArgumentParser(
        description='Train Deep Nash Q-Learning agents for competitive Snake game'
    )
    
    # Training parameters
    parser.add_argument('--train', action='store_true', 
                       help='Train the agents')
    parser.add_argument('--test', action='store_true',
                       help='Test trained agents')
    parser.add_argument('--episodes', type=int, default=3000,
                       help='Number of training episodes (default: 3000)')
    parser.add_argument('--test-episodes', type=int, default=100,
                       help='Number of test episodes (default: 100)')
    
    # Game parameters
    parser.add_argument('--width', type=int, default=15,
                       help='Game board width (default: 15)')
    parser.add_argument('--height', type=int, default=15,
                       help='Game board height (default: 15)')
    
    # Neural network parameters
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden layer size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='Discount factor (default: 0.95)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Mini-batch size (default: 64)')
    parser.add_argument('--buffer-size', type=int, default=50000,
                       help='Replay buffer size (default: 50000)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Initial exploration rate (default: 1.0)')
    parser.add_argument('--epsilon-min', type=float, default=0.01,
                       help='Minimum exploration rate (default: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='Exploration decay rate (default: 0.995)')
    
    # Model parameters
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save/load models (default: models)')
    parser.add_argument('--save-interval', type=int, default=300,
                       help='Save models every N episodes (default: 300)')
    
    args = parser.parse_args()
    
    if not args.train and not args.test:
        # Default: train then test
        args.train = True
        args.test = True
    
    if args.train:
        print("="*70)
        print("DEEP NASH Q-LEARNING TRAINING")
        print("="*70)
        print(f"Episodes: {args.episodes}")
        print(f"Game size: {args.width}x{args.height}")
        print(f"Learning rate: {args.lr}")
        print(f"Discount factor: {args.gamma}")
        print(f"Hidden size: {args.hidden_size}")
        print(f"Batch size: {args.batch_size}")
        print(f"Buffer size: {args.buffer_size}")
        print(f"Epsilon: {args.epsilon} -> {args.epsilon_min} (decay: {args.epsilon_decay})")
        print(f"Model directory: {args.model_dir}")
        print("="*70)
        print()
        
        agent1, agent2 = train_deep_nash_q_learning(
            width=args.width,
            height=args.height,
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            model_dir=args.model_dir,
            learning_rate=args.lr,
            discount_factor=args.gamma,
            batch_size=args.batch_size,
            buffer_capacity=args.buffer_size,
            hidden_size=args.hidden_size
        )
    
    if args.test:
        test_agents(
            model_dir=args.model_dir,
            width=args.width,
            height=args.height,
            num_episodes=args.test_episodes
        )


if __name__ == "__main__":
    main()