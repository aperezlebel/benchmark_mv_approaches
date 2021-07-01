import main
from joblib import Parallel, delayed
from itertools import product


tasks = [
    'TB/death_pvals',
    'TB/platelet_pvals',
    'TB/hemo',
    'TB/hemo_pvals',
    'TB/acid',
    'TB/septic_pvals',
    'UKBB/breast_25',
    'UKBB/breast_pvals',
    'UKBB/skin_pvals',
    'UKBB/parkinson_pvals',
    'UKBB/fluid_pvals',
    'MIMIC/septic_pvals',
    'MIMIC/hemo_pvals',
    'NHIS/income_pvals',
]


def run(task, T):
    argv = [
        'run.py',
        'prediction',
        task,
        '0',
        '--T',
        str(T),
        '--RS',
        '0',
        '--idx',
    ]

    # Only one trial for task having features manually selected (not _pvals)
    if '_pvals' not in task and T != 0:
        return

    main.run(argv)


Parallel(n_jobs=-1)(delayed(run)(task, T) for task, T in product(tasks, range(5)))