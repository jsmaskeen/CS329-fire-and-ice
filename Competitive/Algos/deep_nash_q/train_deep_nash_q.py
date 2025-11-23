"""Train wrapper for Deep Nash Q-Learning

Forwards to the existing `train_deep.py` CLI so the layout matches the
corporative algorithms structure.
"""

from . import train_deep


def main():
    train_deep.main()


if __name__ == '__main__':
    main()
