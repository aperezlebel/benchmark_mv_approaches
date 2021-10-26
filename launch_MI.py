import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=int, default=None, dest='chunk')
parser.add_argument('-s', type=int, default=5, dest='chunk_size')
parser.add_argument('-n', type=int, default=None, dest='train_size')
parser.add_argument('-p', type=str, default=None, dest='partition')
parser.add_argument('-t', type=str, default=None, dest='time')

args = parser.parse_args()


tasks = [
    "TB/death_pvals",
    "TB/platelet_pvals",
    "TB/hemo_pvals",
    "TB/hemo",
    "TB/septic_pvals",
    "UKBB/breast_25",
    "UKBB/breast_pvals",
    "UKBB/skin_pvals",
    "UKBB/parkinson_pvals",
    "UKBB/fluid_pvals",
    "MIMIC/septic_pvals",
    "MIMIC/hemo_pvals",
    "NHIS/income_pvals",
]

reg_tasks = [
    "TB/platelet_pvals",
    "UKBB/fluid_pvals",
    "NHIS/income_pvals",
]

manual_tasks = [
    "TB/hemo",
    "UKBB/breast_25",
]

chunk_size = args.chunk_size
commands = []

for method in [20, 24, 22, 26]:
    for task in tasks:
        if task in reg_tasks and method in [20, 24]:
            continue
        if task not in reg_tasks and method in [22, 26]:
            continue

        db, name = task.split('/')

        for T in range(5):
            train_size_option = '' if args.train_size is None else f' --n {args.train_size}'
            partition_option = '' if args.partition is None else f' --partition {args.partition}'
            time_option = '' if args.time is None else f' --time {args.time}'
            command = f"salloc --ntasks 1 --cpus-per-task 40 --job-name {method}{T}{db[0]}{name}{partition_option}{time_option} srun --pty python main.py predict {task} {method} --RS 0 --T {T} --nbagging 100{train_size_option}"
            session_name = f"{task}_M{method}_T{T}"
            tmux_command = f"tmux new-session -d -s {session_name} '{command}; read'"

            commands.append(tmux_command)


if args.chunk is not None:
    n_tot_chunks = int(np.ceil(len(commands)/chunk_size))
    print(f'\nChunk {args.chunk}/{n_tot_chunks}:\n')

for i, command in enumerate(commands):
    if args.chunk is None or (args.chunk*chunk_size <= i < (args.chunk+1)*chunk_size):
        print(command)


