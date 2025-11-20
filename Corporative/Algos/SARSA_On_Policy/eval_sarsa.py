#!/usr/bin/env python3
"""
Evaluation script for SARSA agent on 2-player Snake game
"""

import os
import sys
import argparse
import numpy as np

# Add current directory to path to enable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sarsa_agent import test_sarsa_agent

def main():
    parser = argparse.ArgumentParser(description='Evaluate SARSA agent for 2-player Snake game')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model file to evaluate')
    parser.add_argument('--width', type=int, default=15, help='Game board width')
    parser.add_argument('--height', type=int, default=15, help='Game board height')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    print(f"Evaluating model: {args.model_path}")
    print(f"Episodes: {args.episodes}")

    scores, lengths = test_sarsa_agent(
        model_path=args.model_path,
        width=args.width,
        height=args.height,
        num_test_episodes=args.episodes
    )

    if scores is None:
        print("Evaluation failed.")
        sys.exit(1)

    # Calculate statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)

    mean_len = np.mean(lengths)
    std_len = np.std(lengths)

    print("\nEvaluation Summary:")
    print(f"Mean Score: {mean_score:.2f} ± {std_score:.2f}")
    print(f"Max Score: {max_score}")
    print(f"Min Score: {min_score}")
    print(f"Mean Episode Length: {mean_len:.2f} ± {std_len:.2f}")

    # Save results to file
    results_file = args.model_path.replace('.pkl', '_eval_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results for {args.model_path}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"Mean Score: {mean_score:.2f} ± {std_score:.2f}\n")
        f.write(f"Max Score: {max_score}\n")
        f.write(f"Min Score: {min_score}\n")
        f.write(f"Mean Episode Length: {mean_len:.2f} ± {std_len:.2f}\n")

    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
