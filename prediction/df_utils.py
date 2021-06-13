"""Functions to manage scores and ranks data frames."""
import numpy as np
import pandas as pd


def assert_equal(s):
    """Check if all value of the series are equal and return the value."""
    if s.empty:
        return 0

    if not (s.iloc[0] == s).all():
        raise ValueError(
            f'Values differ but supposed to be constant. Col: {s.name}.'
        )
    return s.iloc[0]


def aggregate(df, value):
    # Agregate accross folds by averaging
    df['n_folds'] = 1
    dfgb = df.groupby(['size', 'db', 'task', 'method', 'trial'])
    df = dfgb.agg({
        value: 'mean',
        'n_folds': 'sum',
        'scorer': assert_equal,  # first and assert equal
        'selection': assert_equal,
        'n': assert_equal,
        'p': assert_equal,
        'type': assert_equal,
        'imputation_WCT': 'mean',
        'tuning_WCT': 'mean',
        'imputation_PT': 'mean',
        'tuning_PT': 'mean',
    })

    # Agregate accross trials by averaging
    df = df.reset_index()
    df['n_trials'] = 1  # Add a count column to keep track of # of trials
    dfgb = df.groupby(['size', 'db', 'task', 'method'])
    df = dfgb.agg({
        value: 'mean',
        'n_trials': 'sum',
        'n_folds': 'sum',
        'scorer': assert_equal,  # first and assert equal
        'selection': assert_equal,
        'n': assert_equal,
        'p': 'mean',  #assert_equal,
        'type': assert_equal,
        'imputation_WCT': 'mean',
        'tuning_WCT': 'mean',
        'imputation_PT': 'mean',
        'tuning_PT': 'mean',
    })

    # Reset index to addlevel of the multi index to the columns of the df
    df = df.reset_index()

    return df

def get_scores_tab(scores_raw, method_order=None, db_order=None, relative=False, average_sizes=True, formatting=True):
    """Compute article scores tab from raw scores."""
    df = scores_raw.copy()

    if method_order is not None:
        df = df[df['method'].isin(method_order)]

    df = aggregate(df, 'score')
    df.set_index(['size', 'db', 'task', 'method'], inplace=True)

    df = pd.pivot_table(df, values='score', index=['size', 'method'], columns=['db', 'task'])

    if method_order is not None:
        df = df.reindex(method_order, level=1)

    if db_order is not None:
        df = df.reindex(db_order, level=0, axis=1)

    avg_by_size = df.mean(level=0)
    avg_by_size.loc['Global'] = avg_by_size.mean(skipna=True)
    avg_by_size['method'] = 'Reference score'
    avg_by_size.set_index('method', append=True, inplace=True)

    size_order = df.index.get_level_values(0).unique()

    if relative:
        df = df.sub(avg_by_size.droplevel(1, axis=0), level=0)

    if formatting:
        if relative:
            def myround(x):
                if pd.isnull(x):
                    return x
                else:
                    # space = '' if x < 0 else r'\hphantom{-}'
                    # return f'{space}{x:.0e}'
                    s1 = f'{x:.0e}'
                    # Remove the 0 from the exponent
                    if s1[-2] == '0':
                        s2 = s1[:-2] + s1[-1]
                    else:
                        s2 = s1
                    assert np.isclose(float(s1), float(s2)).all()
                    return s2
                # try:
                #     return f'{x:.1e}'
                # except:
                #     return x

            df = df.applymap(myround)

        else:
            df = df.round(3)

    if average_sizes:
        if formatting:
            avg_by_size = avg_by_size.round(3)
        df = pd.concat([df, avg_by_size], axis=0)
        df = df.reindex(list(size_order)+['Global'], level=0)

    def space(x):
        if pd.isnull(x):
            return x
        else:
            space = '' if float(x) < 0 else r'\hphantom{-}'
            return f'{space}{x}'

    if formatting:
        df = df.applymap(space)

    df.index.rename(['Size', 'Method'], inplace=True)
    df.columns.rename(['Database', 'Task'], inplace=True)

    return df


def get_ranks_tab(scores_raw, method_order=None, db_order=None, average_sizes=True):
    """Compute article ranks tab from raw scores."""

    df = scores_raw.copy()

    if method_order is not None:
        df = df[df['method'].isin(method_order)]
    else:
        method_order = df['method'].unique()

    df = aggregate(df, 'score')

    dfgb = df.groupby(['size', 'db', 'task'])
    df['rank'] = dfgb['score'].rank(method='dense', ascending=False)

    df.set_index(['size', 'db', 'task', 'method'], inplace=True)

    df = pd.pivot_table(df, values='rank', index=['size', 'method'], columns=['db', 'task'])

    if method_order is not None:
        df = df.reindex(method_order, level=1)

    if db_order is not None:
        df = df.reindex(db_order, level=0, axis=1)

    if average_sizes:
        avg_on_sizes = df.mean(level=1)
        avg_on_sizes['size'] = 'AVG'
        avg_on_sizes = avg_on_sizes.reset_index().set_index(['size', 'method'])

        df_with_avg_dbs = pd.concat([df, avg_on_sizes], axis=0)

    else:
        df_with_avg_dbs = df

    avg_on_dbs = df_with_avg_dbs.mean(axis=1, level=0)
    avg_on_dbs['All'] = avg_on_dbs.mean(axis=1)

    avg_on_dbs.columns = pd.MultiIndex.from_product([['AVG'], avg_on_dbs.columns])

    def to_int(x):  # Convert to int and robust to NaN
        try:
            return str(int(x))
        except:
            return x


    df = df.applymap(to_int)

    if average_sizes:
        avg_on_sizes = avg_on_sizes.round(1)
        df = pd.concat([df, avg_on_sizes], axis=0)

    avg_on_dbs = avg_on_dbs.round(1)
    df = pd.concat([df, avg_on_dbs], axis=1)

    df.index.rename(['Size', 'Method'], inplace=True)
    df.columns.rename(['Database', 'Task'], inplace=True)

    return df
