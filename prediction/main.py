"""Run the predicitons."""
from .jobs import jobs
from .train import train as train
from .train2 import train as train2
from .train3 import train as train3


def main():
    for task, strategy in jobs:
        _ = train2(task, strategy)
