# Competitive Snake Game with Nash Q-Learning

Two AI snakes compete for the same food using Nash equilibrium strategies.

## 🚀 Quick Start

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

## 🎯 Which Algorithm?

| Feature | Tabular | Deep |
|---------|---------|------|
| **Training Time** | 10 min | 60 min |
| **Best For** | Boards ≤15×15 | Boards ≥20×20 |
| **Complexity** | Simple | Advanced |

**Start with tabular** - it's faster and simpler.

## 🎮 Demo Controls

- **SPACE** - Pause/Resume
- **R** - Restart
- **+/-** - Speed
- **N** - Next episode
- **ESC** - Quit

## 🔧 Advanced Usage

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

## 🐛 Troubleshooting

**Import errors:** Make sure you're in the `competitive/` folder.

**Missing torch:** Run `pip install torch` for deep learning.

**Unbalanced wins:** Normal early on, should reach ~50/50 after training.

## 📊 Expected Results

After full training:
- Win rates: ~50% each
- Avg rewards: 0-5 per episode
- Behavior: Avoids walls, chases food, avoids collisions

---

**Quick commands:**
```bash
python quick_start.py --quick        # Test tabular
python quick_start.py --quick --deep # Test deep
python quick_start.py --train --demo # Train and watch
```
