"""Run the predicitons."""
import logging

from .jobs import jobs, get_job
from .train import train as train
from .train2 import train as train2
from .train3 import train as train3


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.

def main(argv=None):
    if argv is None:
        logger.info('Script executed without argv, reading from jobs.txt.')
        selected_jobs = jobs
    else:
        if len(argv) <= 2:
            raise ValueError('Must pass 0 or 2 arguments in command line.')

        task_name = argv[1]
        strategy_name = argv[2]

        logger.info(f'Argv given. Executing task {task_name} using {strategy_name}.')

        selected_jobs = [get_job(task_name, strategy_name)]

    for task, strategy in selected_jobs:
        _ = train2(task, strategy)
