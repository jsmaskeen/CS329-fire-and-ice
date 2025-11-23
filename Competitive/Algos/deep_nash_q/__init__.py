"""
Deep Nash Q-Learning Algorithm

A deep reinforcement learning variant using neural networks to approximate Q-values
and compute Nash equilibrium strategies. Suitable for large state spaces.
"""

from .deep_nash_qlearning import (
    DQNetwork,
    ReplayBuffer,
    DeepNashQLearningAgent,
    train_deep_nash_q_learning
)

__all__ = [
    'DQNetwork',
    'ReplayBuffer', 
    'DeepNashQLearningAgent',
    'train_deep_nash_q_learning'
]
