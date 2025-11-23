"""Train wrapper for Nash Q-Learning

This small wrapper calls the existing training CLI in `train.py` so that
the competitive algorithms follow the same `train_*.py` layout as the
corporative ones.
"""

from . import train


def main():
    train.main()


if __name__ == '__main__':
    main()
