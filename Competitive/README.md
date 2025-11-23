# Competitive Snake Game

This directory contains the implementation of competitive multi-agent reinforcement learning algorithms for a 2-player snake game.

## Game Environment

The game is played on a grid where two snakes compete for food. The snake that eats the food gets a reward, while the other gets nothing. The game ends when one or both snakes die by hitting a wall or each other.

The state for each agent is represented by a vector of 18 features, including:
-   Danger detection (straight, left, right)
-   Current direction
-   Food direction
-   Opponent's head direction
-   Length comparison with the opponent

## Algorithms

Two competitive algorithms are implemented:

1.  **Nash Q-Learning**: A tabular Q-learning approach extended to a 2-player general-sum game by computing Nash equilibria at each state.
2.  **Deep Nash Q-Learning**: An extension of Nash Q-Learning that uses a deep neural network to approximate the Q-values, allowing it to handle continuous state spaces.

## Running Experiments

To run training and evaluation for all algorithms sequentially, use the `run_all_experiments.ps1` PowerShell script.

```powershell
# Run with default episodes (3000 for training, 100 for testing)
.\run_all_experiments.ps1

# Run with custom episodes
.\run_all_experiments.ps1 -Episodes 5000 -TestEpisodes 200 -SaveInterval 50
```

### Individual Training and Testing

You can also run the training and testing scripts for each algorithm individually.

#### Nash Q-Learning

```bash
# Train the agent
python Competitive/Algos/nash_q/train.py --train --episodes 5000 --save-interval 100

# Test the trained agent
python Competitive/Algos/nash_q/train.py --test --test-episodes 100
```

#### Deep Nash Q-Learning

```bash
# Train the agent
python Competitive/Algos/deep_nash_q/train_deep.py --train --episodes 3000

# Test the trained agent
python Competitive/Algos/deep_nash_q/train_deep.py --test --test-episodes 100
```

### Watching a Demo

To watch a demo of the trained agents playing, you can run the `render.py` or `render_deep.py` scripts.

```bash
# Nash Q-Learning Demo
cd Competitive/Algos/nash_q/
python render.py

# Deep Nash Q-Learning Demo
cd Competitive/Algos/deep_nash_q/
python render_deep.py
```



```ps
PS C:\Users\GSRAJA\Desktop\IIT GN\sem5\fai\CS329-RL\Competitive> .\run_all_experiments.ps1 -Episodes 200 -TestEpisodes 20

========================================================
Processing deep_nash_q...
========================================================
Starting training for 200 episodes...
======================================================================
DEEP NASH Q-LEARNING TRAINING
======================================================================
Episodes: 200
Game size: 15x15
Learning rate: 0.001
Discount factor: 0.95
Hidden size: 128
Batch size: 64
Buffer size: 50000
Epsilon: 1.0 -> 0.01 (decay: 0.995)
Model directory: models
======================================================================

Agent 1 using device: cuda
Agent 2 using device: cuda
No saved model found at models/agent1_deep_nash.pth
No saved model found at models/agent2_deep_nash.pth

Training Deep Nash Q-Learning agents for 200 episodes
Game size: 15x15
Learning rate: 0.001, Discount: 0.95
Batch size: 64, Buffer: 50000
Hidden size: 128
======================================================================
Ep    0 | A1: S= 0 R=   3.2 AvgR=   3.2 Win%=  0.0 Loss= 0.000 | A2: S= 0 R= -11.8 AvgR= -11.8 Win%=  0.0 Loss= 0.000 | Steps= 18 ε=0.995 Buf=   18
Agent 1 saved to models/agent1_deep_nash.pth
Agent 2 saved to models/agent2_deep_nash.pth
Agent 1 saved to models/agent1_deep_nash.pth
Agent 2 saved to models/agent2_deep_nash.pth
Ep   50 | A1: S= 0 R=   4.1 AvgR=  -5.6 Win%=  0.0 Loss= 0.105 | A2: S= 0 R= -10.9 AvgR=  -1.8 Win%= 11.8 Loss= 0.222 | Steps=  9 ε=0.774 Buf=  930
Agent 1 saved to models/agent1_deep_nash.pth
Agent 2 saved to models/agent2_deep_nash.pth
Agent 1 saved to models/agent1_deep_nash.pth
Agent 2 saved to models/agent2_deep_nash.pth
Ep  100 | A1: S= 0 R= -10.9 AvgR=  -4.8 Win%=  3.0 Loss= 0.156 | A2: S= 1 R=  14.1 AvgR=  -2.1 Win%= 12.9 Loss= 0.282 | Steps=  9 ε=0.603 Buf= 1924
Agent 1 saved to models/agent1_deep_nash.pth
Agent 2 saved to models/agent2_deep_nash.pth
Agent 1 saved to models/agent1_deep_nash.pth
Agent 2 saved to models/agent2_deep_nash.pth
Agent 1 saved to models/agent1_deep_nash.pth
Agent 2 saved to models/agent2_deep_nash.pth
Ep  150 | A1: S= 0 R=   3.5 AvgR=  -4.6 Win%=  3.3 Loss= 0.208 | A2: S= 0 R= -11.5 AvgR=  -2.4 Win%= 10.6 Loss= 0.320 | Steps= 15 ε=0.469 Buf= 2917
Agent 1 saved to models/agent1_deep_nash.pth
Agent 2 saved to models/agent2_deep_nash.pth
Agent 1 saved to models/agent1_deep_nash.pth
Agent 2 saved to models/agent2_deep_nash.pth
Agent 1 saved to models/agent1_deep_nash.pth
Agent 2 saved to models/agent2_deep_nash.pth

======================================================================
Training completed!
Agent 1 - Wins: 6, Losses: 23, Draws: 171
Agent 2 - Wins: 23, Losses: 6, Draws: 171
Replay buffer sizes - Agent1: 3948, Agent2: 3948
======================================================================
Starting testing for 20 episodes...

Testing trained Deep Nash Q-Learning agents for 20 episodes...
Agent 1 using device: cuda
Agent 2 using device: cuda
Agent 1 loaded from models/agent1_deep_nash.pth (hidden_size=128)
Agent 2 loaded from models/agent2_deep_nash.pth (hidden_size=128)
Test Episode   0 | Agent1:  0 | Agent2:  0 | Steps:   8
Test Episode  10 | Agent1:  0 | Agent2:  0 | Steps:  29

============================================================
DEEP NASH Q-LEARNING TEST RESULTS
============================================================
Agent 1 Wins: 1 (5.0%)
Agent 2 Wins: 2 (10.0%)
Draws: 17 (85.0%)

Agent 1 Avg Score: 0.05 ± 0.22
Agent 2 Avg Score: 0.10 ± 0.30

Average Episode Length: 20.4 ± 11.1
============================================================

========================================================
Processing nash_q...
========================================================
Starting training for 200 episodes...
============================================================
NASH Q-LEARNING TRAINING
============================================================
Episodes: 200
Game size: 15x15
Learning rate: 0.1
Discount factor: 0.95
Epsilon: 1.0 -> 0.01 (decay: 0.995)
Model directory: models
============================================================

No saved model found at models/agent1_nash.pkl
No saved model found at models/agent2_nash.pkl
Training Nash Q-Learning agents for 200 episodes
Game size: 15x15
Learning rate: 0.1, Discount: 0.95
Episode     0 | Agent1: Score= 0 Reward= -11.1 AvgR= -11.1 Win%=  0.0 | Agent2: Score= 0 Reward=   3.9 AvgR=   3.9 Win%=  0.0 | Steps= 11 ε=0.995 Q-states=    7
Agent 1 saved to models/agent1_nash.pkl
Agent 2 saved to models/agent2_nash.pkl
Agent 1 saved to models/agent1_nash.pkl
Agent 2 saved to models/agent2_nash.pkl
Agent 1 saved to models/agent1_nash.pkl
Agent 2 saved to models/agent2_nash.pkl
Agent 1 saved to models/agent1_nash.pkl
Agent 2 saved to models/agent2_nash.pkl
Episode   100 | Agent1: Score= 1 Reward=  -3.1 AvgR=  -6.1 Win%=  6.9 | Agent2: Score= 0 Reward=   1.9 AvgR=  -1.5 Win%=  5.9 | Steps= 31 ε=0.603 Q-states=  277
Agent 1 saved to models/agent1_nash.pkl
Agent 2 saved to models/agent2_nash.pkl
Agent 1 saved to models/agent1_nash.pkl
Agent 2 saved to models/agent2_nash.pkl
Agent 1 saved to models/agent1_nash.pkl
Agent 2 saved to models/agent2_nash.pkl
Agent 1 saved to models/agent1_nash.pkl
Agent 2 saved to models/agent2_nash.pkl
Agent 1 saved to models/agent1_nash.pkl
Agent 2 saved to models/agent2_nash.pkl
Agent 1 saved to models/agent1_nash.pkl
Agent 2 saved to models/agent2_nash.pkl

Training completed!
Agent 1 - Wins: 13, Losses: 15, Draws: 172
Agent 2 - Wins: 15, Losses: 13, Draws: 172
Q-table sizes - Agent1: 371, Agent2: 328
Starting testing for 20 episodes...

Testing trained agents for 20 episodes...
Agent 1 loaded from models/agent1_nash.pkl
Agent 2 loaded from models/agent2_nash.pkl
Test Episode   0 | Agent1:  0 | Agent2:  0 | Steps:   8
Test Episode  10 | Agent1:  0 | Agent2:  0 | Steps:  15

============================================================
TEST RESULTS SUMMARY
============================================================
Agent 1 Wins: 1 (5.0%)
Agent 2 Wins: 3 (15.0%)
Draws: 16 (80.0%)

Agent 1 Avg Score: 0.10 ± 0.30
Agent 2 Avg Score: 0.20 ± 0.40

Average Episode Length: 25.9 ± 19.6
============================================================

All competitive experiments completed.
```