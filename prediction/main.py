"""Run the predicitons."""
import logging
import argparse

from .strategies import strategies
from .tasks import tasks
from .train4 import train


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

# Parser config
parser = argparse.ArgumentParser(description='Prediction program.')
parser.add_argument('program')
parser.add_argument('task_name', nargs='?', default=None)
parser.add_argument('strategy_name', nargs='?', default=None)
parser.add_argument('--RS', dest='RS', default=None, nargs='?',
                    help='The random state to use.')
parser.add_argument('--T', dest='T', default=0, nargs='?',
                    help='The trial #.')
parser.add_argument('--n_top_pvals', dest='n_top_pvals', default=100, nargs='?',
                    help='The trial #.')


def run(argv=None):
    """Train the choosen model(s) on the choosen task(s)."""
    args = parser.parse_args(argv)

    task_name = args.task_name
    strategy_name = args.strategy_name
    RS = args.RS
    T = args.T
    n_top_pvals = args.n_top_pvals

    # Try to convert to int if id passed
    try:
        strategy_name = int(strategy_name)
    except ValueError:  # If error, then it's a name and not an id.
        pass

    if isinstance(strategy_name, int):
        strategy_name = list(strategies.keys())[strategy_name]

    task = tasks.get(task_name, RS=RS, T=T, n_top_pvals=n_top_pvals)
    strategy = strategies[strategy_name]

    logger.info(f'Run task {task_name} using {strategy_name}')
    logger.info(f'Asked RS {RS} T {T} n_top_pvals {n_top_pvals}')

    if RS:
        RS = int(RS)

    train(task, strategy, RS=RS, T=T)
