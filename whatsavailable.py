"""Helper to knwo what strategies and tasks are availables."""
from prediction.tasks import tasks
from prediction.strategies import strategies


def strategies_available():
    print('\nModels available:')
    for i, name in enumerate(strategies.keys()):
        print(f'\t{i}: '+name)
    print()


def tasks_available():
    print('\nTasks available:')
    for name in tasks.keys():
        print('\t'+name)
    print()


def run(args):
    if args.method:
        strategies_available()
    if args.task:
        tasks_available()
    if not args.method and not args.task:
        strategies_available()
        tasks_available()
