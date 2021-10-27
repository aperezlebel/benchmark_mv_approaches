"""Run the predicitons."""
import logging

from .strategies import strategies
from .tasks import tasks
from .train import train
from .PlotHelper import PlotHelper


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.


def run(args):
    task_name = args.task_name
    strategy_name = args.strategy_name
    RS = args.RS
    T = args.T
    n_top_pvals = args.n_top_pvals
    dump_idx_only = args.dump_idx_only

    # Try to convert to int if id passed
    try:
        strategy_name = int(strategy_name)
    except ValueError:  # If error, then it's a name and not an id.
        pass

    if isinstance(strategy_name, int):
        strategy_name = list(strategies.keys())[strategy_name]

    task = tasks.get(task_name, RS=RS, T=T, n_top_pvals=n_top_pvals)
    strategy = strategies[strategy_name]

    logger.info(f'Run task {task_name} using {strategy_name}')
    logger.info(f'Asked RS {RS} T {T} n_top_pvals {n_top_pvals}')

    if RS:
        RS = int(RS)

    train(task, strategy, RS=RS, T=T, dump_idx_only=dump_idx_only,
          n_bagging=args.n_bagging, train_size=args.train_size,
          n_permutation=args.n_permutation)


rename = {
    '': 'MIA',
    '_imputed_Mean': 'Mean',
    '_imputed_Mean+mask': 'Mean+mask',
    '_imputed_Med': 'Med',
    '_imputed_Med+mask': 'Med+mask',
    '_imputed_Iterative': 'Iter',
    '_imputed_Iterative+mask': 'Iter+mask',
    '_imputed_KNN': 'KNN',
    '_imputed_KNN+mask': 'KNN+mask',
    '_imputed_Iterative_Bagged100': 'MI',
    '_imputed_Iterative+mask_Bagged100': 'MI+mask',
    '_Logit_imputed_Mean': 'Linear+Mean',
    '_Logit_imputed_Mean+mask': 'Linear+Mean+mask',
    '_Logit_imputed_Med': 'Linear+Med',
    '_Logit_imputed_Med+mask': 'Linear+Med+mask',
    '_Logit_imputed_Iterative': 'Linear+Iter',
    '_Logit_imputed_Iterative+mask': 'Linear+Iter+mask',
    '_Logit_imputed_KNN': 'Linear+KNN',
    '_Logit_imputed_KNN+mask': 'Linear+KNN+mask',
    '_Ridge_imputed_Mean': 'Linear+Mean',
    '_Ridge_imputed_Mean+mask': 'Linear+Mean+mask',
    '_Ridge_imputed_Med': 'Linear+Med',
    '_Ridge_imputed_Med+mask': 'Linear+Med+mask',
    '_Ridge_imputed_Iterative': 'Linear+Iter',
    '_Ridge_imputed_Iterative+mask': 'Linear+Iter+mask',
    '_Ridge_imputed_KNN': 'Linear+KNN',
    '_Ridge_imputed_KNN+mask': 'Linear+KNN+mask',
}


def aggregate_results(args):
    ph = PlotHelper(root_folder=args.root_folder, rename=rename)
    ph.dump('scores/test_scores.csv')
