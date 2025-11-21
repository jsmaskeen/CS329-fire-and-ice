#!/usr/bin/env python3
"""
Simple test script for competitive snake game
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from Game.game import CompetitiveSnakeGame

def test_game():
    """Test basic game functionality"""
    print("Testing Competitive Snake Game...")
    print("=" * 50)
    
    # Create game
    game = CompetitiveSnakeGame(10, 10)
    print("✓ Game created successfully")
    
    # Test initial state
    state1, state2 = game.get_state()
    print(f"✓ State vectors: Agent1={len(state1)} dims, Agent2={len(state2)} dims")
    
    # Test a few steps
    print("\nTesting game steps:")
    for i in range(5):
        r1, r2, (s1, s2), done, info = game.step(0, 0)
        print(f"  Step {i+1}: R1={r1:5.1f}, R2={r2:5.1f}, Done={done}, "
              f"Scores=({info['snake1_score']}, {info['snake2_score']})")
        
        if done:
            print(f"  Game ended at step {i+1}")
            break
    
    print("\n✓ All tests passed!")
    print("\nTo train agents, run:")
    print("  python competitive/train.py --train --episodes 1000")
    print("\nTo watch trained agents play:")
    print("  python competitive/render.py")

if __name__ == "__main__":
    try:
        test_game()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)