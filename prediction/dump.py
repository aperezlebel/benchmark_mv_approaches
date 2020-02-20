"""Function to dump task info and fit results into csv/yaml files."""
import pandas as pd
import os
import yaml
# from datetime import datetime

results_folder = 'results/'

def dump_infos(item, filepath):
    """Dump the infos at the given place.

    Parameters
    ----------
    item : object
        Object implementing  a get_infos method returning the infos to dump.
    filepath : string
        Path to the place to store the infos.
    """
    data = item.get_infos()
    with open(filepath, 'w') as file:
        file.write(yaml.dump(data, allow_unicode=True))


def dump_task_strategy(task, strategy):
    """Dump the infos of the task and strategy used."""
    meta = task.meta
    task_folder = f'{results_folder}{meta.db}/{task.count}_{meta.name}/'
    strategy_folder = f'{task_folder}{strategy.count}_{strategy.estimator_class()}/'

    os.makedirs(strategy_folder, exist_ok=True)

    dump_infos(task, f'{task_folder}task_infos.yml')
    dump_infos(strategy, f'{strategy_folder}strat_infos.yml')
