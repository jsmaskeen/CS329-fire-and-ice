"""
Nash Q-Learning (Tabular) Algorithm

A multi-agent reinforcement learning algorithm that learns Nash equilibrium strategies
using Q-tables. Suitable for smaller state spaces.
"""

from .nash_qlearning import NashQLearningAgent, train_nash_q_learning

__all__ = ['NashQLearningAgent', 'train_nash_q_learning']
