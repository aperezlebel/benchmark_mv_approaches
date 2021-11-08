"""Main script. Configure logger and argparser. Launch scripts"""
import os
import argparse
import logging
import time

import prediction
import selection
import statistics
import pvals
import whatsavailable
import whosmissing
import dump_ids


if __name__ == '__main__':
    # Configure logger
    logs_folder = 'logs/'
    os.makedirs(logs_folder, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())  # Print also in console.
    log_filepath = f'{logs_folder}{int(1e6*time.time())}_prediction.log'
    logging.basicConfig(
        filename=log_filepath,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d:%(levelname)s:%(module)s.%(funcName)s:'
        '%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Configure parser
    parser = argparse.ArgumentParser(description='main program')
    subparsers = parser.add_subparsers(dest='action')

    # Script 1: Compute pvals for feature selection with ANOVA
    p = subparsers.add_parser('select', description='Compute p-values for '
                              'feature selection with ANOVA')
    p.set_defaults(func=selection.run)
    p.add_argument('task_name', nargs='?', default=None)
    p.add_argument('--RS', dest='RS', default=0, nargs='?',
                   help='The random state to use.')
    p.add_argument('--T', dest='T', default=0, nargs='?',
                   help='The trial #.')
    p.add_argument('--TMAX', dest='TMAX', default=5, nargs='?',
                   help='The max # of trials.')

    # Script 2: Filter p-values
    p = subparsers.add_parser('filter', description='Filter all p-values.')
    p.set_defaults(func=pvals.filter)

    # Script 3: Run experiments
    p = subparsers.add_parser('predict', description='Launch experiment for '
                              '1 task, 1 method and 1 trial.')
    p.set_defaults(func=prediction.run)
    p.add_argument('task_name', default=None, help='Name of the '
                   'task.')
    p.add_argument('strategy_name', default=None, help='Name or id of the '
                   'method. See `python main.py info available` for ids.')
    p.add_argument('--RS', dest='RS', default=0, nargs='?',
                   help='The random state to use.')
    p.add_argument('--T', dest='T', default=0, nargs='?',
                   help='The trial #.')
    p.add_argument('--n_top_pvals', dest='n_top_pvals', default=100,
                   nargs='?', help='The trial #.')
    p.add_argument('--idx', dest='dump_idx_only', default=False, const=True,
                   nargs='?', help='Dump only the idx (no prediction).')
    p.add_argument('--nbagging', type=int, default=None, dest='n_bagging')
    p.add_argument('--n', type=int, default=None, dest='train_size')
    p.add_argument('--npermutation', type=int, default=None, dest='n_permutation')
    p.add_argument('--fold', type=int, default=None, dest='asked_fold')
    p.add_argument('--out', type=str, default=None, dest='results_folder')

    # Script 4: Aggregate results
    p = subparsers.add_parser('aggregate', description='Aggregate results.')
    p.set_defaults(func=prediction.aggregate_results)
    p.add_argument('--root', type=str, help='The root folder where the '
                   'results are stored.', default='results/', dest='root_folder')
    p.add_argument('-n', type=int, default=None, dest='n')
    p.add_argument('--out', type=str, default='test_scores', dest='out')

    # Script 5: Figures and tables of the paper
    p = subparsers.add_parser('figs', description='Build figure and tables '
                              'of the paper.')
    p.set_defaults(func=statistics.run)
    subp = p.add_subparsers(dest='action')
    parent_l = argparse.ArgumentParser(add_help=False)
    parent_l.add_argument('--linear', dest='linear', default=False, const=True,
                          nargs='?', help='Whether to use linear methods')
    parent_csv = argparse.ArgumentParser(add_help=False)
    parent_csv.add_argument('--csv', dest='csv', default=False, const=True,
                            nargs='?',
                            help='Whether to dump into csv as well.')

    p = subp.add_parser('wilcoxon', parents=[parent_l, parent_csv],
                        description='Wilcoxon test.')
    p.add_argument('--less', type=bool, default=False, const=True, nargs='?',
                   help='Whether to use greater or less one sided wilcoxon.')

    subp.add_parser('friedman', parents=[parent_l, parent_csv],
                    description='Friedman & Nemenyi tests + crit. difference.')

    subp.add_parser('scores', parents=[parent_l, parent_csv],
                    description='Compute scores and ranks.')

    p = subp.add_parser('boxplot', parents=[parent_l],
                        description='Plot boxplots scores & times.')

    p = subp.add_parser('desc', description='Overview table.')

    p = subp.add_parser('time', description='Total time of fit.')

    p = subp.add_parser('difficulty', description='Plot rank vs difficulty.')
    p.add_argument('--no-avg', type=bool, nargs='?', const=False, default=True, dest='avg')

    p = subp.add_parser('breakout', parents=[parent_l],
                        description='Plot broken out boxplots scores & times.')

    p = subp.add_parser('mi', description='Plot multiple imputation results.')
    p.add_argument('-n', type=int, default=None)

    # Script 6: Data statistics
    p = subparsers.add_parser('datastats', description='Build figures and '
                              'tables of the paper on data statistics.')
    p.set_defaults(func=statistics.run)
    subp = p.add_subparsers(dest='action')

    p = subp.add_parser('mv', help='Missing values distributions.')
    p.add_argument('tag', default=None, nargs='?', help='The task tag')
    p.add_argument('--hide', dest='hide', default=False, const=True,
                   nargs='?', help='Whether to plot the stats or print')
    p.add_argument('--fig1', dest='fig1', default=False, const=True,
                   nargs='?', help='Whether to plot the figure1')
    p.add_argument('--fig2', dest='fig2', default=False, const=True,
                   nargs='?', help='Whether to plot the figure2')
    p.add_argument('--fig2b', dest='fig2b', default=False, const=True,
                   nargs='?', help='Whether to plot the figure2')
    p.add_argument('--fig3', dest='fig3', default=False, const=True,
                   nargs='?', help='Whether to plot the figure3')

    p = subp.add_parser('ftypes', help='Feature types')

    p = subp.add_parser('fcor', parents=[parent_csv],
                        help='Feature correlations.')
    p.add_argument('--t', type=float, default=0.1,
                   help='Threshold for correlation')
    p.add_argument('--abs', type=bool, default=False, const=True, nargs='?',
                   help='Whether to use absolute values of correlation')

    # Script 7: Get information
    p = subparsers.add_parser('info', description='Get informations.')
    subp = p.add_subparsers(dest='action')
    p = subp.add_parser('available',
                        description='What task/method is available.')
    p.add_argument('-t', dest='task', type=bool, default=False, const=True,
                   nargs='?', help='Get task info.')
    p.add_argument('-m', dest='method', type=bool, default=False, const=True,
                   nargs='?', help='Get method info.')
    p.set_defaults(func=whatsavailable.run)
    p = subp.add_parser('missing', description='Who is missing in scores.')
    p.set_defaults(func=whosmissing.run)

    # Script 8: Dump IDs
    p = subparsers.add_parser('ids', description='Dump all IDs used.')
    p.set_defaults(func=dump_ids.run)

    # Start run
    logger.info('Started run')
    print(f'Dumping logs into {log_filepath}')

    args = parser.parse_args()
    args.func(args)

    logger.info('Ended run')
