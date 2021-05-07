import argparse
import os
import matplotlib.pyplot as plt

from .tests import run_friedman, run_wilcoxon, run_scores
from .statistics import run_mv, run_prop
from custom.const import local_graphics_folder, remote_graphics_folder


plt.rcParams.update({
    'text.usetex': True,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'axes.labelsize': 15,
    'legend.fontsize': 11,
    'legend.title_fontsize': 11,
    'figure.figsize': (8, 4.8),
    # 'figure.dpi': 600,
})


def run(argv=None):
    parser = argparse.ArgumentParser(description='Stats on missing values.')
    parser.add_argument('program')

    subparsers = parser.add_subparsers(dest='action')
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

    subparsers.add_parser('wilcoxon', parents=[parent_a, parent_csv])
    subparsers.add_parser('friedman', parents=[parent_a, parent_l, parent_csv])
    subparsers.add_parser('scores', parents=[parent_a, parent_l, parent_csv])
    p = subparsers.add_parser('mv', parents=[parent_a])
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

    p = subparsers.add_parser('prop', parents=[parent_a])
    p.add_argument('tag', default=None, nargs='?', help='The task tag')

    args = parser.parse_args(argv)

    graphics_folder = local_graphics_folder
    if args.article:
        graphics_folder = remote_graphics_folder
    os.makedirs(graphics_folder, exist_ok=True)
    print(f'Dump into "{graphics_folder}"')

    if args.action == 'wilcoxon':
        run_wilcoxon(graphics_folder, csv=args.csv)

    elif args.action == 'friedman':
        run_friedman(graphics_folder, linear=args.linear, csv=args.csv)

    elif args.action == 'scores':
        run_scores(graphics_folder, linear=args.linear, csv=args.csv)

    elif args.action == 'mv':
        run_mv(args, graphics_folder)

    elif args.action == 'prop':
        run_prop(args, graphics_folder)

    else:
        raise ValueError(f'Not known action {args.action}.')
