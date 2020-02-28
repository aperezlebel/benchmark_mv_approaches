"""Run the predicitons."""
from .jobs import jobs
from .train import train


def main():
    for task, strategy in jobs:
        _ = train(task, strategy)
