"""Run the predicitons."""
import logging
import argparse

from .jobs import jobs, get_job
from .train2 import train as train2
from .train3 import train as train3
from .train4 import train as train4


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
parser.add_argument('--train4', dest='train', const=train4, default=train2,
                    nargs='?',
                    help='Whether to use train2 or train4 for prediction.')
parser.add_argument('--RS', dest='RS', default=None, nargs='?',
                    help='The random state to use.')


def run(argv=None):
    """Train the choosen model(s) on the choosen task(s)."""
    args = parser.parse_args(argv)

    task_name = args.task_name
    strategy_name = args.strategy_name
    RS = args.RS

    if task_name is None or strategy_name is None:
        logger.info('No task or strategy given. Reading from jobs.txt.')
        selected_jobs = jobs
    else:
        logger.info(f'Argv given. Task: {task_name} ; Strategy name/id: {strategy_name}')

        # Try to convert to int if id passed
        try:
            strategy_name = int(strategy_name)
        except ValueError: # If error, then it's a name and not an id.
            pass

        selected_jobs = [get_job(task_name, strategy_name)]

        logger.info(f'Run task {task_name} using {selected_jobs[0][1].name}')

    logger.info(f'Asked RS: {RS}')

    for task, strategy in selected_jobs:
        _ = args.train(task, strategy, RS=RS)
