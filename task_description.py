"""Get some information on all the tasks."""
import os
import pandas as pd
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


def plot_index_sizes():
    """Plot the stored index sizes. must be computed before."""
    if not os.path.exists(index_sizes_path):
        raise ValueError(f'No index sizes found at {index_sizes_path}')

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_title('Size of the indexes of the prediction tasks (after removing '
                 'indexes used for pvals if any)')
    df = pd.read_csv(index_sizes_path, index_col=0)
    df = df.sort_values(by='index_size', ascending=False)
    df = df.dropna(subset=['index_size'])
    sns.set_color_codes('muted')
    sns.barplot(x='index_size', y='tag', data=df, ax=ax, color='b')

    print(df)
    plt.show()



if __name__ == '__main__':
    # compute_index_sizes()
    plot_index_sizes()


