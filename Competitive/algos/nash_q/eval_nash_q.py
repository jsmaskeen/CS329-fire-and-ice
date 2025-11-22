"""Evaluation wrapper for Nash Q-Learning

Provides a small CLI to run the test routine implemented in `train.py`.
"""

from .train import test_agents

import argparse


def main():
    parser = argparse.ArgumentParser(description='Evaluate Nash Q agents')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--width', type=int, default=15)
    parser.add_argument('--height', type=int, default=15)
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()

    test_agents(model_dir=args.model_dir, width=args.width, height=args.height, num_episodes=args.episodes)


if __name__ == '__main__':
    main()
