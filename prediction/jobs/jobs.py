"""Load jobs from job file if any."""
import os

from ..strategies import strategies
from ..tasks import tasks

jobs = []

# Loading jobs from file
filepath = 'custom/jobs.txt'

if os.path.exists(filepath):
    with open(filepath, 'r') as file:
        for line in file.read().splitlines():
            task_name, startegy_name = line.split(' ')
            jobs.append((tasks[task_name], strategies[startegy_name]))
