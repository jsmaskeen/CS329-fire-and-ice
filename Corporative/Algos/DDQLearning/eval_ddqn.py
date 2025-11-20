#!/usr/bin/env python3
"""
Evaluation script for DDQN agent on 2-player Snake game
"""

import os
import sys
import argparse
import numpy as np

# Add current directory to path to enable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ddqn_agent import test_ddqn_agent

def main():
    parser = argparse.ArgumentParser(description='Evaluate DDQN agent for 2-player Snake game')
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
    
    scores, lengths = test_ddqn_agent(
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
    
    # Prepare report
    report = []
    report.append("=" * 50)
    report.append(f"Evaluation Report")
    report.append("=" * 50)
    report.append(f"Model Path: {os.path.abspath(args.model_path)}")
    report.append(f"Episodes: {args.episodes}")
    report.append(f"Board Size: {args.width}x{args.height}")
    report.append("-" * 50)
    report.append(f"Score Mean: {mean_score:.2f}")
    report.append(f"Score Std:  {std_score:.2f}")
    report.append(f"Score Max:  {max_score}")
    report.append(f"Score Min:  {min_score}")
    report.append("-" * 50)
    report.append(f"Length Mean: {mean_len:.2f}")
    report.append(f"Length Std:  {std_len:.2f}")
    report.append("=" * 50)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report to the same directory as the model
    model_dir = os.path.dirname(args.model_path)
    report_path = os.path.join(model_dir, 'eval_results.txt')
    
    with open(report_path, 'w') as f:
        f.write(report_text)
        
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    main()
