#!/usr/bin/env python3
"""
Quick start script for competitive Nash Q-Learning snake game
"""

import sys
import os
import argparse

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

def main():
    parser = argparse.ArgumentParser(
        description='Competitive Snake Game with Nash Q-Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (100 episodes)
  python quick_start.py --quick
  
  # Full training (5000 episodes)
  python quick_start.py --train
  
  # Watch trained agents play
  python quick_start.py --demo
  
  # Train and then demo
  python quick_start.py --train --demo
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 100 episodes')
    parser.add_argument('--train', action='store_true',
                       help='Full training with 5000 episodes')
    parser.add_argument('--demo', action='store_true',
                       help='Watch trained agents play')
    parser.add_argument('--deep', action='store_true',
                       help='Use Deep Nash Q-Learning (neural networks)')
    parser.add_argument('--episodes', type=int,
                       help='Custom number of training episodes')
    
    args = parser.parse_args()
    
    if not any([args.quick, args.train, args.demo]):
        parser.print_help()
        return
    
    # Determine algorithm type
    algo_name = "Deep Nash Q-Learning" if args.deep else "Nash Q-Learning"
    model_suffix = "deep_nash" if args.deep else "nash"
    
    # Quick test
    if args.quick:
        print("\n" + "="*60)
        print(f"QUICK TEST - {algo_name} - 100 episodes")
        print("="*60 + "\n")
        
        if args.deep:
            from algos.deep_nash_q import train_deep_nash_q_learning
            train_deep_nash_q_learning(
                width=10,
                height=10,
                num_episodes=100,
                save_interval=50,
                model_dir='models',
                hidden_size=64  # Smaller for quick test
            )
        else:
            from algos.nash_q import train_nash_q_learning
            train_nash_q_learning(
                width=10,
                height=10,
                num_episodes=100,
                save_interval=50,
                model_dir='models'
            )
        
        print(f"\nQuick test completed! Run with --demo{' --deep' if args.deep else ''} to watch agents play.")
        return
    
    # Full training
    if args.train:
        episodes = args.episodes if args.episodes else (3000 if args.deep else 5000)
        
        print("\n" + "="*60)
        print(f"TRAINING - {algo_name} - {episodes} episodes")
        print("="*60 + "\n")
        
        if args.deep:
            from algos.deep_nash_q import train_deep_nash_q_learning
            train_deep_nash_q_learning(
                width=15,
                height=15,
                num_episodes=episodes,
                save_interval=300,
                model_dir='models',
                hidden_size=128
            )
        else:
            from algos.nash_q import train_nash_q_learning
            train_nash_q_learning(
                width=15,
                height=15,
                num_episodes=episodes,
                save_interval=500,
                model_dir='models'
            )
        
        print(f"\nTraining completed! Models saved in models/")
    
    # Demo
    if args.demo:
        print("\n" + "="*60)
        print(f"DEMO - {algo_name} - Watching trained agents play")
        print("="*60 + "\n")
        
        try:
            if args.deep:
                from algos.deep_nash_q.render_deep import run_demo
            else:
                from algos.nash_q.render import run_demo
                
            run_demo(
                model_dir='models',
                width=15,
                height=15,
                num_episodes=5,
                cell_size=30,
                speed=8
            )
        except ImportError as e:
            if 'pygame' in str(e):
                print("ERROR: pygame is required for demo")
                print("Install with: pip install pygame")
            elif 'torch' in str(e):
                print("ERROR: pytorch is required for Deep Nash Q-Learning")
                print("Install with: pip install torch")
            else:
                raise

if __name__ == "__main__":
    main()