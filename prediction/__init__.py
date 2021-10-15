"""Run the predicitons."""
import logging

from .strategies import strategies
from .tasks import tasks
from .train import train
from .PlotHelper import PlotHelper


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.


def run(args):
    task_name = args.task_name
    strategy_name = args.strategy_name
    RS = args.RS
    T = args.T
    n_top_pvals = args.n_top_pvals
    dump_idx_only = args.dump_idx_only

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

    train(task, strategy, RS=RS, T=T, dump_idx_only=dump_idx_only,
          n_bagging=args.n_bagging)


def aggregate_results(args):
    ph = PlotHelper(root_folder='results')
    ph.dump('scores/test_scores.csv')
