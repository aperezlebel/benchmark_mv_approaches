"""Load jobs from job file if any."""
import os

from ..strategies import strategies
from ..tasks import tasks


def get_job(task_name, strategy_name):
    return (tasks[task_name], strategies[strategy_name])


jobs = []

# Loading jobs from file
filepath = 'custom/jobs.txt'

if os.path.exists(filepath):
    with open(filepath, 'r') as file:
        for line in file.read().splitlines():
            task_name, strategy_name = line.split(' ')
            jobs.append(get_job(task_name, strategy_name))

