"""Load jobs from job file if any."""
import os

from ..strategies import strategies
from ..tasks import tasks


def get_job(task_name, strategy_name):
    """Load the wanted task and strategy.

    Parameters
    ----------
    task_name : str
        Name of the task with format DB/prediction_task
    strategy_name : str or int
        If str: Name of the strategy. ex: Classification.
        If int: Id of the strategy.

    Returns
    -------
    tuple
        Size 2 tuple with Task and Strategy object.

    """
    if isinstance(strategy_name, int):
        strategy_name = list(strategies.keys())[strategy_name]

    return (tasks[task_name], strategies[strategy_name])


jobs = []

# Loading jobs from file
filepath = 'custom/jobs.txt'

if os.path.exists(filepath):
    with open(filepath, 'r') as file:
        for line in file.read().splitlines():
            task_name, strategy_name = line.split(' ')
            jobs.append(get_job(task_name, strategy_name))

