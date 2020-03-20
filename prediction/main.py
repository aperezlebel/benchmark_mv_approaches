"""Run the predicitons."""
import logging
import argparse

from .jobs import jobs, get_job
from .train2 import train as train2
from .train3 import train as train3


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

# Parser config
parser = argparse.ArgumentParser(description='Prediction program.')
parser.add_argument('program')
parser.add_argument('task_name', nargs='?', default=None)
parser.add_argument('strategy_name', nargs='?', default=None)
parser.add_argument('--train3', dest='train', const=train3, default=train2,
                    nargs='?',
                    help='Whether to use train2 or train3 for prediction.')


def run(argv=None):
    """Train the choosen model(s) on the choosen task(s)."""
    args = parser.parse_args(argv)

    task_name = args.task_name
    strategy_name = args.strategy_name

    if task_name is None or strategy_name is None:
        logger.info('No task or strategy given. Reading from jobs.txt.')
        selected_jobs = jobs
    else:
        logger.info(f'Argv given. Run task {task_name} using {strategy_name}.')
        selected_jobs = [get_job(task_name, strategy_name)]

    for task, strategy in selected_jobs:
        _ = args.train(task, strategy)
