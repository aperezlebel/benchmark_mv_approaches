import argparse
import os
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=int, default=None, dest='chunk')
parser.add_argument('-s', type=int, default=5, dest='chunk_size')
parser.add_argument('-n', type=int, default=None, dest='train_size')
parser.add_argument('-p', type=str, default=None, dest='partition')
parser.add_argument('-t', type=str, default=None, dest='time')
parser.add_argument('-f', type=str, default=None, dest='fold')
parser.add_argument('--cpu', type=int, default=40, dest='n_cpus')
parser.add_argument('-m', type=str, choices=['mi', 'mia', 'mmean'], default='mi', dest='method')
parser.add_argument('--mem', type=str, default=None, dest='memory')
parser.add_argument('-a', type=str, default=None, dest='account')
parser.add_argument('--run', type=bool, nargs='?', default=False, const=True, dest='run')
parser.add_argument('--nbagging', type=int, default=100, dest='n_bagging')
parser.add_argument('--npermutation', type=int, default=None, dest='n_permutation')
parser.add_argument('--no-slurm', type=bool, nargs='?', default=True, const=False, dest='slurm')
parser.add_argument('--out', type=str, default=None, dest='results_folder')

args = parser.parse_args()


if args.n_bagging == 0:
    args.n_bagging = None

if args.n_permutation == 0:
    args.n_permutation = None

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

sizes_max = {
    "TB/death_pvals": 10000,
    "TB/platelet_pvals": 10000,
    "TB/hemo_pvals": 10000,
    "TB/hemo": 10000,
    "TB/septic_pvals": 2500,
    "UKBB/fluid_pvals": 25000,
    "MIMIC/septic_pvals": 25000,
    "MIMIC/hemo_pvals": 25000,
    "NHIS/income_pvals": 10000,
}

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

methods = [20, 24, 22, 26] if args.method == 'mi' else [0, 2]

reg_methods = [22, 26] if args.method == 'mi' else [2]
clf_methods = [20, 24] if args.method == 'mi' else [0]

if args.method == 'mi':
    methods = [20, 24, 22, 26]
    reg_methods = [22, 26]
    clf_methods = [20, 24]

elif args.method == 'mia':
    methods = [0, 2]
    reg_methods = [2]
    clf_methods = [0]

elif args.method == 'mmean':
    methods = [8, 10]
    reg_methods = [10]
    clf_methods = [8]

python_path = sys.executable

for method in methods:
    for task in tasks:
        if task in reg_tasks and method in clf_methods:
            continue
        if task not in reg_tasks and method in reg_methods:
            continue

        n_max = sizes_max.get(task, None)
        if n_max is not None and args.train_size is not None and int(args.train_size) > n_max:
            continue

        db, name = task.split('/')

        for T in range(5):
            train_size_option = '' if args.train_size is None else f' --n {args.train_size}'
            bagging_option = '' if args.n_bagging is None else f' --nbagging {args.n_bagging}'
            permutation_option = '' if args.n_permutation is None else f' --npermutation {args.n_permutation}'
            partition_option = '' if args.partition is None else f' --partition {args.partition}'
            time_option = '' if args.time is None else f' --time {args.time}'
            memory_option = '' if args.memory is None else f' --mem {args.memory}'
            account_option = '' if args.account is None else f' --account {args.account}'
            out_option = '' if args.results_folder is None else f' --out {args.results_folder}'
            fold_option = '' if args.fold is None else f' --fold {args.fold}'
            slurm_command = f"salloc --ntasks 1 --cpus-per-task {args.n_cpus} --job-name {method}{T}{db[0]}{name}{partition_option}{time_option}{memory_option}{account_option} srun --pty " if args.slurm else ''
            command = f"{slurm_command}{python_path} main.py predict {task} {method} --RS 0 --T {T} {bagging_option}{permutation_option}{train_size_option}{out_option}{fold_option}"
            session_name = f"{task}_n{args.train_size}_bag{args.n_bagging}_perm{args.n_permutation}_M{method}_T{T}"
            tmux_command = f"tmux new-session -d -s {session_name} '{command}; read'"

            commands.append(tmux_command)

			# Break for tasks that don't need 5 trials
            if task in manual_tasks:
                break


if args.chunk is not None:
    n_tot_chunks = int(np.ceil(len(commands)/chunk_size))
    print(f'\nChunk {args.chunk + 1}/{n_tot_chunks}:\n')

for i, command in enumerate(commands):
    if args.chunk is None or (args.chunk*chunk_size <= i < (args.chunk+1)*chunk_size):
        print(command)
        print()
        if args.run:
            os.system(command)
        print()


