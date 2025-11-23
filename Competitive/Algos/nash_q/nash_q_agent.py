"""Agent wrapper for Nash Q-Learning

This module exposes the `NashQLearningAgent` class from the existing
`nash_qlearning` implementation so other code can import a consistent
`*_agent.py` module like the corporative algorithms.
"""

from .nash_qlearning import NashQLearningAgent

__all__ = ["NashQLearningAgent"]
