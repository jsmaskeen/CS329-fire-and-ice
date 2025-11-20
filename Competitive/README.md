# Competitive Snake Game with Nash Q-Learning

Two AI snakes compete for the same food using Nash equilibrium strategies.

## Quick Start

### Installation
```bash
# Required
pip install numpy scipy pygame

# For Deep Nash Q-Learning
pip install torch
```

### Run Algorithms

**Nash Q-Learning (Tabular):**
```bash
cd competitive

# Quick test (100 episodes, ~30 seconds)
python quick_start.py --quick

# Full training (5000 episodes, ~10 minutes)
python quick_start.py --train

# Watch trained agents
python quick_start.py --demo
```

**Deep Nash Q-Learning:**
```bash
# Quick test (100 episodes, ~2 minutes)
python quick_start.py --quick --deep

# Full training (3000 episodes, ~60 minutes)
python quick_start.py --train --deep

# Watch trained agents
python quick_start.py --demo --deep
```

### Custom Training

**Tabular:**
```bash
cd algos/nash_q
python train.py --episodes 10000 --lr 0.15
```

**Deep:**
```bash
cd algos/deep_nash_q
python train_deep.py --episodes 5000 --hidden-size 256 --batch-size 128
```

### Test Models
```bash
# Tabular
cd algos/nash_q
python train.py --test

# Deep
cd algos/deep_nash_q
python train_deep.py --test
```

## 📁 Structure

```
competitive/
├── quick_start.py       # Main entry point
├── game.py              # Game engine
├── algos/
│   ├── nash_q/          # Tabular algorithm
│   └── deep_nash_q/     # Deep learning algorithm
└── models/              # Saved models
```

**Quick commands:**
```bash
python quick_start.py --quick        # Test tabular
python quick_start.py --quick --deep # Test deep
python quick_start.py --train --demo # Train and watch
```
