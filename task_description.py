"""Get some information on all the tasks."""
import yaml
import os
import pandas as pd
from copy import copy
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import seaborn as sns

from prediction.tasks import tasks

index_sizes_dir = 'sandbox/'
index_sizes_path = f'{index_sizes_dir}index_sizes.csv'


def compute_index_sizes():
    """Compute index sizes of all tasks."""
    rows = []

    for tag in tasks.keys():
        db, name = tag.split('/')
        print(tag)
        try:
            if db == 'UKBB':
                raise AssertionError  # skip for testing
            task = tasks.get(tag, n_top_pvals=100)
            y = task.y
            s = y.shape[0]
            print(f'\t{s}')
        except AssertionError:
            s = None
            print(f'\tpvals file not found')

        rows.append([tag, db, name, s])

    df = pd.DataFrame(rows, columns=['tag', 'db', 'name', 'index_size'])
    os.makedirs(index_sizes_dir, exist_ok=True)
    df.to_csv(index_sizes_path)


def plot_index_sizes(min_test_size):
    """Plot the stored index sizes. must be computed before."""
    if not os.path.exists(index_sizes_path):
        raise ValueError(f'No index sizes found at {index_sizes_path}')

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_title('Size of the indexes of the prediction tasks (after removing '
                 'indexes used for pvals if any)')
    df = pd.read_csv(index_sizes_path, index_col=0)
    df = df.sort_values(by='index_size', ascending=False)
    df = df.dropna(subset=['index_size'])

    df['max_train_size'] = df['index_size']*(1-min_test_size)
    print(df)

    sns.set_color_codes('pastel')
    sns.barplot(x='index_size', y='tag', data=df, ax=ax, color='lightgrey',
                label=f'Min test size ({100*min_test_size:.1f} %)')
    patches_to_avoid = copy(ax.patches)
    sns.set_color_codes('muted')
    sns.barplot(x='max_train_size', y='tag', data=df, ax=ax, color='b',
                label='Max train size')

    for p in ax.patches:
        if p in patches_to_avoid:
            continue
        _x = p.get_x() + p.get_width()/2
        _y = p.get_y() + p.get_height()/2
        value = int(p.get_width())
        ax.text(_x, _y, value, ha='right', color='white')

    plt.legend()
    plt.show()



if __name__ == '__main__':
    plot_index_sizes(min_test_size=0.1)


