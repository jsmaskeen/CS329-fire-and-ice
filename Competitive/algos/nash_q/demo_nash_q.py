"""Demo wrapper for Nash Q-Learning

This module forwards to the existing visual demo in `render.py` so it
matches the `demo_*.py` pattern used in the corporative algorithms.
"""

from . import render


def main():
    render.main()


if __name__ == "__main__":
    main()
