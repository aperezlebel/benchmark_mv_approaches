"""Main script. Configure logger, load .env."""
import argparse
import logging
import os
import random
import string

import extraction
import prediction
import selection
import statistics

# Configure logger
logs_folder = 'logs/'

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.


def get_and_increase_count():
    count_filepath = logs_folder+'count.txt'

    if not os.path.exists(count_filepath):
        count = 0
    else:
        with open(count_filepath, 'r') as file:
            c = file.read()
            if c == '':
                count = 0
            else:
                count = int(c) + 1

    # Dump new count
    os.makedirs(logs_folder, exist_ok=True)
    with open(count_filepath, 'w') as file:
        file.write(str(count))

    return count


count = get_and_increase_count()


def get_log_filepath(filename):
    slug = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    return f'{logs_folder}{count}_{slug}_{filename}'


log_filepath = get_log_filepath('prediction.log')

logging.basicConfig(
    filename=log_filepath,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d:%(levelname)s:%(module)s.%(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Configure parser
parser = argparse.ArgumentParser(description='main program')
subparsers = parser.add_subparsers(dest='action')

# Prediction script
p = subparsers.add_parser('predict')
p.set_defaults(func=prediction.run)
p.add_argument('task_name', nargs='?', default=None)
p.add_argument('strategy_name', nargs='?', default=None)
p.add_argument('--RS', dest='RS', default=None, nargs='?',
               help='The random state to use.')
p.add_argument('--T', dest='T', default=0, nargs='?',
               help='The trial #.')
p.add_argument('--n_top_pvals', dest='n_top_pvals', default=100,
               nargs='?', help='The trial #.')
p.add_argument('--idx', dest='dump_idx_only', default=False, const=True,
               nargs='?', help='Dump only the idx (no prediction).')

# Select features
p = subparsers.add_parser('select')
p.set_defaults(func=selection.run)
p.add_argument('task_name', nargs='?', default=None)
p.add_argument('--RS', dest='RS', default=0, nargs='?',
               help='The random state to use.')
p.add_argument('--T', dest='T', default=0, nargs='?',
               help='The trial #.')
p.add_argument('--TMAX', dest='TMAX', default=5, nargs='?',
               help='The max # of trials.')

# Extract
p = subparsers.add_parser('extract')
p.set_defaults(func=extraction.run)
p.add_argument('task_name', nargs='?', default=None)
p.add_argument('--RS', dest='RS', default=None, nargs='?',
               help='The random state to use.')

# Statistics
p = subparsers.add_parser('stats')
p.set_defaults(func=statistics.run)
subp = p.add_subparsers(dest='action')
parent_l = argparse.ArgumentParser(add_help=False)
parent_l.add_argument(
    '--l', dest='linear', default=False, const=True,
    nargs='?', help='Whether to use linear methods')
parent_a = argparse.ArgumentParser(add_help=False)
parent_a.add_argument(
    '--a', dest='article', default=False, const=True,
    nargs='?', help='Whether to dump into article folder.')
parent_csv = argparse.ArgumentParser(add_help=False)
parent_csv.add_argument(
    '--csv', dest='csv', default=False, const=True,
    nargs='?', help='Whether to dump into csv as well.')

p = subp.add_parser('wilcoxon', parents=[parent_a, parent_l, parent_csv])
p.add_argument('--less', type=bool, default=False, const=True, nargs='?',
               help='Whether to use greater or less one sided wilcoxon.')

subp.add_parser('friedman', parents=[parent_a, parent_l, parent_csv])

subp.add_parser('scores', parents=[parent_a, parent_l, parent_csv])

p = subp.add_parser('mv', parents=[parent_a])
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

p = subp.add_parser('prop', parents=[parent_a])
p.add_argument('tag', default=None, nargs='?', help='The task tag')

p = subp.add_parser('cor', parents=[parent_a, parent_csv])
p.add_argument('--t', type=float, default=0.1,
               help='Threshold for correlation')
p.add_argument('--abs', type=bool, default=False, const=True, nargs='?',
               help='Whether to use absolute values of correlation')

p = subp.add_parser('boxplot', parents=[parent_a, parent_l])
p = subp.add_parser('desc', parents=[parent_a])
p = subp.add_parser('time', parents=[parent_a])

args = parser.parse_args()

# Start run
logger.info('Started run')
print(f'Dumping logs into {log_filepath}')

args = parser.parse_args()
args.func(args)

logger.info('Ended run')
