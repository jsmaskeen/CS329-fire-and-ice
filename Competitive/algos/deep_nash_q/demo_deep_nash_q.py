"""Demo wrapper for Deep Nash Q-Learning visualiser

Forwards to the existing `render_deep.py` demo so it matches the
`demo_*.py` pattern used elsewhere in the project.
"""

from . import render_deep


def main():
    render_deep.main()


if __name__ == '__main__':
    main()
