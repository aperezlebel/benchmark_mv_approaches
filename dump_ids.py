from argparse import Namespace
from itertools import product

from joblib import Parallel, delayed

import prediction


def run(args):
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

    def run_one(task, T):
        argv = {
            'action': 'prediction',
            'task_name': task,
            'strategy_name': '0',
            'T': str(T),
            'RS': '0',
            'dump_idx_only': True,
            'n_top_pvals': 100,
        }

        # Only one trial for task having features manually selected (not _pvals)
        if '_pvals' not in task and T != 0:
            return

        args = Namespace(**argv)
        prediction.run(args)

    Parallel(n_jobs=-1)(delayed(run_one)(task, T) for task, T in product(tasks, range(5)))
