"""Helper to knwo what strategies and tasks are availables."""
import sys

from prediction.tasks import tasks
from prediction.strategies import strategies


def strategies_available():
    print('\nStrategies available:')
    for i, name in enumerate(strategies.keys()):
        print(f'\t{i}: '+name)
    print()


def tasks_available():
    print('\nTasks available:')
    for name in tasks.keys():
        print('\t'+name)
    print()


if 'strategies' in sys.argv:
    strategies_available()
elif 'tasks' in sys.argv:
    tasks_available()
elif len(sys.argv) == 1:
    strategies_available()
    tasks_available()
else:
    raise ValueError('Unrecognized argument(s).')
