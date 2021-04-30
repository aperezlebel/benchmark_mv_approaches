"""Functions to manage scores and ranks data frames."""
import pandas as pd

from .PlotHelper import PlotHelper

def get_scores_tab(scores_raw, method_order=None, db_order=None):
    """Compute article scores tab from raw scores."""
    df = scores_raw.copy()

    if method_order is not None:
        df = df[df['method'].isin(method_order)]

    df = PlotHelper.aggregate(df, 'score')
    df.set_index(['size', 'db', 'task', 'method'], inplace=True)

    df = pd.pivot_table(df, values='score', index=['size', 'method'], columns=['db', 'task'])

    if method_order is not None:
        df = df.reindex(method_order, level=1)

    if db_order is not None:
        df = df.reindex(db_order, level=0, axis=1)

    avg_by_size = df.mean(level=0)
    avg_by_size.loc['Global'] = avg_by_size.mean(skipna=True)
    avg_by_size['method'] = 'AVG'
    avg_by_size.set_index('method', append=True, inplace=True)

    size_order = df.index.get_level_values(0).unique()

    df = pd.concat([df, avg_by_size], axis=0)

    df = df.reindex(list(size_order)+['Global'], level=0)

    df = df.round(2)

    return df
