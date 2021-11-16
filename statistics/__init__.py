import os
import matplotlib.pyplot as plt

from custom.const import local_graphics_folder, remote_graphics_folder
from .tests import run_friedman, run_wilcoxon
from .statistics import run_mv, run_prop, run_cor, run_time
from .boxplots import run_boxplot
from .tabs import run_desc, run_scores
from .difficulty import run_difficulty
from .breakout import run_breakout
from .mi import run_multiple_imputation
from .importance import run_feature_importance


plt.rcParams.update({
    'text.usetex': True,
    'mathtext.fontset': 'stix',
    'font.family': 'STIXGeneral',
    'axes.labelsize': 15,
    'legend.fontsize': 11,
    'legend.title_fontsize': 11,
    'figure.figsize': (8, 4.8),
})


def run(args):
    graphics_folder = local_graphics_folder
    if hasattr(args, 'article') and args.article:
        graphics_folder = remote_graphics_folder
    os.makedirs(graphics_folder, exist_ok=True)
    print(f'Dump into "{graphics_folder}"')

    if args.action == 'wilcoxon':
        run_wilcoxon(graphics_folder, linear=args.linear,
                     csv=args.csv, greater=not args.less)

    elif args.action == 'friedman':
        run_friedman(graphics_folder, linear=args.linear, csv=args.csv, ref=args.ref)

    elif args.action == 'scores':
        run_scores(graphics_folder, linear=args.linear, csv=args.csv)

    elif args.action == 'mv':
        run_mv(args, graphics_folder)

    elif args.action == 'ftypes':
        run_prop(args, graphics_folder)

    elif args.action == 'fcor':
        run_cor(args, graphics_folder, csv=args.csv, absolute=args.abs)

    elif args.action == 'boxplot':
        run_boxplot(graphics_folder, linear=args.linear)

    elif args.action == 'desc':
        run_desc(graphics_folder)

    elif args.action == 'time':
        run_time()

    elif args.action == 'difficulty':
        run_difficulty(graphics_folder, args.avg)

    elif args.action == 'breakout':
        run_breakout(graphics_folder, linear=args.linear)

    elif args.action == 'mi':
        run_multiple_imputation(graphics_folder, args.n, args.bagging_only, args.linear)

    elif args.action == 'imp':
        run_feature_importance(graphics_folder, args.root, args.n,
                               args.average_folds, args.mode, args.hue_by_task)

    else:
        raise ValueError(f'Not known action {args.action}.')
