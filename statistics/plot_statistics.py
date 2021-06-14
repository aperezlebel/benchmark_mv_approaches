"""Plot statistis about missing values from given indicators."""
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost


# Plot functions: each indicator has a different way of being plotted
def plot_global(indicators, plot=False, show=True, ax=None):
    """Plot statistics on the full database."""
    # Get required indicators
    df = indicators['global']

    n_rows = df.at[0, 'n_rows']
    n_cols = df.at[0, 'n_cols']
    n_values = df.at[0, 'n_values']
    n_mv = df.at[0, 'n_mv']
    n_mv1 = df.at[0, 'n_mv1']
    n_mv2 = df.at[0, 'n_mv2']
    n_not_mv = df.at[0, 'n_not_mv']
    f_mv = df.at[0, 'f_mv']
    f_mv1 = df.at[0, 'f_mv1']
    f_mv2 = df.at[0, 'f_mv2']
    f_not_mv = df.at[0, 'f_not_mv']

    # Print these statistics
    if show:
        print(
            f'\n'
            f'Statistics on the full data frame:\n'
            f'---------------------------------\n'
            f'[{n_rows} rows x {n_cols} columns]\n'
            f'{n_values} values\n'
            f'N NMV:    {f_not_mv:.1f}% or {n_not_mv}\n'
            f'N MV:     {f_mv:.1f}% or {n_mv}\n'
            f'    N MV 1:   {f_mv1:.1f}% or {n_mv1}\n'
            f'    N MV 2:   {f_mv2:.1f}% or {n_mv2}\n'
        )

    # If asked, plot these statistics
    if plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        df_show = pd.DataFrame({
            'MV1': [n_mv1],
            'MV2': [n_mv2],
            'MV': [n_mv],
            'V': [n_values],
            'type': ['Full data frame']
            })

        sns.set_color_codes('pastel')
        sns.barplot(x='V', y='type', data=df_show, color='lightgray', ax=ax,
                    label=f'Not missing ({f_not_mv:.1f}%)')

        sns.set_color_codes('muted')
        sns.barplot(x='MV', y='type', data=df_show, color='b', ax=ax,
                    label=f'Missing - Not applicable ({f_mv1:.1f}%)')

        sns.set_color_codes('dark')
        sns.barplot(x='MV2', y='type', data=df_show, color='b', ax=ax,
                    label=f'Missing - Not available ({f_mv2:.1f}%)')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.5, box.height*0.5])

        ax.legend(ncol=1, loc='center left', frameon=True,
                  title='Type of values',
                  bbox_to_anchor=(1.05, 0.5))
        ax.set(ylabel='', xlabel=f'Number of values (Total {n_values})')
        ax.set_title('Proportion of missing values')
        sns.despine(left=True, bottom=True, ax=ax)

        # Remove y labels
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)


def plot_features(indicators, plot=False, show=True, ax=None):
    """Plot the number of features with missing values."""
    # Get required indicators
    df = pd.concat([indicators['features'], indicators['global']], axis=1)

    n_f_w_mv = df.at[0, 'n_f_w_mv']
    n_f_w_mv1_o = df.at[0, 'n_f_w_mv1_o']
    n_f_w_mv2_o = df.at[0, 'n_f_w_mv2_o']
    n_f_w_mv_1a2 = df.at[0, 'n_f_w_mv_1a2']
    n_f_wo_mv = df.at[0, 'n_f_wo_mv']
    f_f_w_mv = df.at[0, 'f_f_w_mv']
    f_f_w_mv1_o = df.at[0, 'f_f_w_mv1_o']
    f_f_w_mv2_o = df.at[0, 'f_f_w_mv2_o']
    f_f_w_mv_1a2 = df.at[0, 'f_f_w_mv_1a2']
    f_f_wo_mv = df.at[0, 'f_f_wo_mv']

    n_cols = df.at[0, 'n_cols']

    if show:
        print(
            f'\n'
            f'Statistics on features:\n'
            f'-----------------------\n'
            f'N features: {n_cols}\n'
            f'N features with MV:              {n_f_w_mv} ({f_f_w_mv:.1f}%)\n'
            f'    N features with MV1 only:    {n_f_w_mv1_o} ({f_f_w_mv1_o:.1f}%)\n'
            f'    N features with MV2 only:    {n_f_w_mv2_o} ({f_f_w_mv2_o:.1f}%)\n'
            f'    N features with MV1 and MV2: {n_f_w_mv_1a2} ({f_f_w_mv_1a2:.1f}%)\n'
        )

    if plot:
        # Plot proportion of features with missing values
        df_show = pd.DataFrame({
            'N MV': [n_f_w_mv],
            'N MV1 only': [n_f_w_mv1_o],
            'N MV2 only': [n_f_w_mv2_o],
            'N MV 1 xor 2': [n_f_w_mv1_o + n_f_w_mv2_o],
            'N F': [n_cols],
            'type': ['Full data frame']
            })

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        sns.set_color_codes('pastel')
        sns.barplot(x='N F', y='type', data=df_show, color='lightgray', ax=ax,
                    label=f'No missing values ({n_f_wo_mv} • {f_f_wo_mv:.1f}%)')

        sns.set_color_codes('pastel')
        sns.barplot(x='N MV', y='type', data=df_show, color='g', ax=ax,
                    label=f'Not applicable and not available ({n_f_w_mv_1a2} • {f_f_w_mv_1a2:.1f}%)')

        sns.set_color_codes('muted')
        sns.barplot(x='N MV 1 xor 2', y='type', data=df_show, color='g', ax=ax,
                    label=f'Not applicable only ({n_f_w_mv1_o} • {f_f_w_mv1_o:.1f}%)')

        sns.set_color_codes('dark')
        sns.barplot(x='N MV2 only', y='type', data=df_show, color='g', ax=ax,
                    label=f'Not available only ({n_f_w_mv2_o} • {f_f_w_mv2_o:.1f}%)')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.5, box.height*0.5])

        ax.legend(ncol=1, loc='center left', frameon=True,
                  title='Type of missing values contained in the feature',
                  bbox_to_anchor=(1.05, 0.5))
        ax.set(ylabel='', xlabel=f'Number of features (Total {n_cols})')
        ax.set_title('Proportion of features having missing values')
        sns.despine(left=True, bottom=True, ax=ax)

        # Remove y labels
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)


def plot_feature_wise(indicators, plot=False, show=True, ax=None, nf_max=40):
    """Plot the statistics feature-wise."""
    n_mv_fw = indicators['feature-wise']

    n_rows = indicators['global'].at[0, 'n_rows']

    if show:
        with pd.option_context('display.max_rows', None):
            print(
                f'\n'
                f'Statistics feature-wise:\n'
                f'------------------------\n'
                f'\n'
                f'{n_mv_fw}'
            )

    if plot:
        # Plot proportion of missing values in each feature
        # Copy index in a column for the barplot method
        n_mv_fw['feature'] = n_mv_fw.index
        n_mv_fw['feature_shortened'] = n_mv_fw['id'].astype(str) + ': ' + n_mv_fw.index

        # Truncate
        if n_mv_fw.shape[0] <= nf_max:
            def truncate(string):
                if len(string) <= 20:
                    return string
                return string[:27]+'...'

            n_mv_fw['feature_shortened'] = n_mv_fw['feature_shortened'].apply(truncate)

        # Add the total number of values for each feature
        n_mv_fw['N V'] = n_rows

        # Get rid of the features with no missing values
        n_mv_fw_l = n_mv_fw[(n_mv_fw['N MV1'] != 0) | (n_mv_fw['N MV2'] != 0)]

        n_mv_fw_l = n_mv_fw_l.head(20)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = plt.gcf()

        if n_mv_fw_l.empty:
            return fig, ax

        sns.set_color_codes('pastel')
        sns.barplot(x='N V', y='feature_shortened', data=n_mv_fw_l, ax=ax,
                    color='lightgray', label=f'Not missing', dodge=False)

        sns.set_color_codes('muted')
        sns.barplot(x='N MV', y='feature_shortened', data=n_mv_fw_l, ax=ax,
                    color='b', label=f'Missing - Not applicable')

        sns.set_color_codes("dark")
        sns.barplot(x='N MV2', y='feature_shortened', data=n_mv_fw_l, ax=ax,
                    color="b", label=f'Missing - Not available')

        ax.legend(ncol=1, loc='lower right', frameon=True,
                  title='Type of values')
        ax.set(ylabel='Features', xlabel='Number of values')
        ax.tick_params(labelsize=7)
        sns.despine(left=True, bottom=True, ax=ax)

        # Remove y labels if more than 40
        if n_mv_fw_l.shape[0] > nf_max:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            fig.tight_layout(rect=(0, 0, 1, .92))

        else:
            fig.tight_layout(rect=(0., 0, 1, .92))

        return fig, ax


def plot_feature_wise_v2(indicators, plot=False, show=True, ax=None, nf_max=40, color='b'):
    """Plot the statistics feature-wise."""
    n_mv_fw = indicators['feature-wise']

    if show:
        with pd.option_context('display.max_rows', None):
            print(
                f'\n'
                f'Statistics feature-wise:\n'
                f'------------------------\n'
                f'\n'
                f'{n_mv_fw}'
            )

    if plot:
        # Plot proportion of missing values in each feature
        # Copy index in a column for the barplot method
        n_mv_fw['feature'] = n_mv_fw.index

        n_mv_fw['id'] = np.arange(n_mv_fw.shape[0])

        # Get rid of the features with no missing values
        n_mv_fw_l = n_mv_fw

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = plt.gcf()

        sns.set_color_codes('pastel')
        handle_nm, = ax.stackplot(n_mv_fw_l['id'].values, 100, color='lightgray', labels=['Not missing'])
        handle_m, = ax.stackplot(n_mv_fw_l['id'].values, n_mv_fw_l['F MV'].values, color=color, labels=['Missing'])
        # ax.stackplot(n_mv_fw_l['id'].values, n_mv_fw_l['N V'].values, color='lightgray', labels=['Not missing'])
        # ax.stackplot(n_mv_fw_l['id'].values, n_mv_fw_l['N MV'].values, color='b', labels=['Missing'])

        # ax.legend(ncol=1, loc='upper right', frameon=True,
        #           title='Type of values')
        # ax.set(xlabel='Features', ylabel='Proportion')
        # ax.set(xlabel='Features', ylabel='Number of values')
        ax.tick_params(labelsize=7)
        sns.despine(left=True, bottom=True, ax=ax)

        # Remove y labels if more than 40
        if n_mv_fw.shape[0] > nf_max:
            fig.tight_layout(rect=(0, 0, 1, .92))
        else:
            fig.tight_layout(rect=(0., 0, 1, .92))

        return fig, ax, (handle_nm, handle_m)


def plot_rows(indicators, plot=False, show=True, ax=None):
    """Plot stats on rows without missing values."""
    # Get required indicators
    df = pd.concat([indicators['rows'], indicators['global']], axis=1)

    n_r_wo_mv = df.at[0, 'n_r_wo_mv']
    n_r_w_mv = df.at[0, 'n_r_w_mv']
    n_r_w_mv1_o = df.at[0, 'n_r_w_mv1_o']
    n_r_w_mv2_o = df.at[0, 'n_r_w_mv2_o']
    n_r_w_mv_1a2 = df.at[0, 'n_r_w_mv_1a2']
    f_r_wo_mv = df.at[0, 'f_r_wo_mv']
    f_r_w_mv = df.at[0, 'f_r_w_mv']
    f_r_w_mv1_o = df.at[0, 'f_r_w_mv1_o']
    f_r_w_mv2_o = df.at[0, 'f_r_w_mv2_o']
    f_r_w_mv_1a2 = df.at[0, 'f_r_w_mv_1a2']

    n_rows = df.at[0, 'n_rows']

    if show:
        print(
            f'\n'
            f'Statistics on rows:\n'
            f'-------------------\n'
            f'N rows: {n_rows}\n'
            f'N rows without MV:         {n_r_wo_mv} ({f_r_wo_mv:.2f}%)\n'
            f'N rows with MV:            {n_r_w_mv} ({f_r_w_mv:.2f}%)\n'
            f'  N rows with MV1 only:    {n_r_w_mv1_o} ({f_r_w_mv1_o:.2f}%)\n'
            f'  N rows with MV2 only:    {n_r_w_mv2_o} ({f_r_w_mv2_o:.2f}%)\n'
            f'  N rows with MV1 and MV2: {n_r_w_mv_1a2} ({f_r_w_mv_1a2:.2f}%)\n'
        )

    if plot:
        # Plot proportion of features with missing values
        df_show = pd.DataFrame({
            'N MV': [n_r_w_mv],
            'N MV1 only': [n_r_w_mv1_o],
            'N MV2 only': [n_r_w_mv2_o],
            'N MV 1 xor 2': [n_r_w_mv1_o + n_r_w_mv2_o],
            'N R': [n_rows],
            'type': ['Full data frame']
            })

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        sns.set_color_codes('pastel')
        sns.barplot(x='N R', y='type', data=df_show, color='lightgray', ax=ax,
                    label=f'No missing values ({n_r_wo_mv} • {f_r_wo_mv:.2f}%)')

        sns.set_color_codes('pastel')
        sns.barplot(x='N MV', y='type', data=df_show, color='r', ax=ax,
                    label=f'Not applicable and not available ({n_r_w_mv_1a2} • {f_r_w_mv_1a2:.2f}%)')

        sns.set_color_codes('muted')
        sns.barplot(x='N MV 1 xor 2', y='type', data=df_show, color='r', ax=ax,
                    label=f'Not applicable only ({n_r_w_mv1_o} • {f_r_w_mv1_o:.2f}%)')

        sns.set_color_codes('dark')
        sns.barplot(x='N MV2 only', y='type', data=df_show, color='r', ax=ax,
                    label=f'Not available only ({n_r_w_mv2_o} • {f_r_w_mv2_o:.2f}%)')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.5, box.height*0.5])

        ax.legend(ncol=1, loc='center left', frameon=True,
                  title='Type of missing values contained in the row',
                  bbox_to_anchor=(1.05, 0.5))
        ax.set(ylabel='', xlabel=f'Number of rows (Total {n_rows})')
        ax.set_title('Proportion of rows having missing values')
        sns.despine(left=True, bottom=True, ax=ax)

        # Remove y labels
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)


def plot_rm_rows(indicators, plot=False, show=True, ax=None):
    """Plot number of rows affected if we remove features with MV."""
    # Get required indicators
    df = pd.concat([indicators['rm_rows'], indicators['global']], axis=1)

    n_r_a_rm_mv1 = df.at[0, 'n_r_a_rm_mv1']
    n_r_a_rm_mv2 = df.at[0, 'n_r_a_rm_mv2']
    n_r_a_rm_mv_1o2 = df.at[0, 'n_r_a_rm_mv_1o2']
    n_r_a_rm_mv1_o = df.at[0, 'n_r_a_rm_mv1_o']
    n_r_a_rm_mv2_o = df.at[0, 'n_r_a_rm_mv2_o']
    n_r_a_rm_mv_1a2 = df.at[0, 'n_r_a_rm_mv_1a2']
    f_r_a_rm_mv1 = df.at[0, 'f_r_a_rm_mv1']
    f_r_a_rm_mv2 = df.at[0, 'f_r_a_rm_mv2']
    f_r_a_rm_mv_1o2 = df.at[0, 'f_r_a_rm_mv_1o2']
    f_r_a_rm_mv1_o = df.at[0, 'f_r_a_rm_mv1_o']
    f_r_a_rm_mv2_o = df.at[0, 'f_r_a_rm_mv2_o']
    f_r_a_rm_mv_1a2 = df.at[0, 'f_r_a_rm_mv_1a2']

    n_rows = df.at[0, 'n_rows']

    if show:
        print(
            f'N rows losing information if we remove features with :\n'
            f'    MV1:          {n_r_a_rm_mv1} ({f_r_a_rm_mv1:.2f}%)\n'
            f'    MV2:          {n_r_a_rm_mv2} ({f_r_a_rm_mv2:.2f}%)\n'
            f'    MV:           {n_r_a_rm_mv_1o2} ({f_r_a_rm_mv_1o2:.2f}%)\n'
            f'    MV1 only:     {n_r_a_rm_mv1_o} ({f_r_a_rm_mv1_o:.2f}%)\n'
            f'    MV2 only:     {n_r_a_rm_mv2_o} ({f_r_a_rm_mv2_o:.2f}%)\n'
            f'    MV1 and MV2:  {n_r_a_rm_mv_1a2} ({f_r_a_rm_mv_1a2:.2f}%)\n'
        )

    if plot:
        # Plot number of rows losing information when removing features with MV
        df_show = pd.DataFrame({
            'N rows affected': [
                n_r_a_rm_mv1,
                n_r_a_rm_mv2,
                n_r_a_rm_mv_1o2,
                # n_r_a_rm_mv1_o,
                # n_r_a_rm_mv2_o,
                # n_r_a_rm_mv_1a2,
            ],
            'N R': [n_rows for _ in range(3)],
            # 'N R': [n_rows for _ in range(6)],
            'type': [
                f'Not applicable\n{f_r_a_rm_mv1:.2f}%',
                f'Not available\n{f_r_a_rm_mv2:.2f}%',
                f'Not applicable or\nnot available\n{f_r_a_rm_mv_1o2:.2f}%',
                # f'MV1 only\n{f_r_a_rm_mv1_o:.2f}%',
                # f'MV2 only\n{f_r_a_rm_mv2_o:.2f}%',
                # f'MV1 and MV2\n{f_r_a_rm_mv_1a2:.2f}%'
            ],
        })

        df_show.sort_values('N rows affected', ascending=False, inplace=True)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        sns.set_color_codes('muted')
        sns.barplot(x='N rows affected', y='type', data=df_show, color='r', ax=ax)

        box = ax.get_position()
        ax.set_position([1.1*box.x0, box.y0, box.width, box.height])

        ax.set_title('Number of rows losing information (non-missing values)\n'
                     'when removing features containing missing values of type:')
        ax.set(ylabel='', xlabel=f'Number of rows (Total {n_rows})')
        ax.set_xlim(right=n_rows)
        sns.despine(left=True, bottom=True, ax=ax)


def plot_rm_features(indicators, plot=False, show=True, ax=None):
    """Plot the part of information lost when removing features with MV."""
    # Get required indicators
    df = pd.concat([indicators['rm_features'], indicators['global']], axis=1)

    n_v_lost_mv1 = df.at[0, 'n_v_lost_mv1']
    n_v_lost_mv2 = df.at[0, 'n_v_lost_mv2']
    n_v_lost_mv_1o2 = df.at[0, 'n_v_lost_mv_1o2']
    n_v_lost_mv1_o = df.at[0, 'n_v_lost_mv1_o']
    n_v_lost_mv2_o = df.at[0, 'n_v_lost_mv2_o']
    n_v_lost_mv_1a2 = df.at[0, 'n_v_lost_mv_1a2']
    f_v_lost_mv1 = df.at[0, 'f_v_lost_mv1']
    f_v_lost_mv2 = df.at[0, 'f_v_lost_mv2']
    f_v_lost_mv_1o2 = df.at[0, 'f_v_lost_mv_1o2']
    f_v_lost_mv1_o = df.at[0, 'f_v_lost_mv1_o']
    f_v_lost_mv2_o = df.at[0, 'f_v_lost_mv2_o']
    f_v_lost_mv_1a2 = df.at[0, 'f_v_lost_mv_1a2']

    n_rows = df.at[0, 'n_rows']
    n_values = df.at[0, 'n_values']

    if show:
        print(
            f'N values lost if we remove features with :\n'
            f'    MV1:          {n_v_lost_mv1} ({f_v_lost_mv1:.2f}%)\n'
            f'    MV2:          {n_v_lost_mv2} ({f_v_lost_mv2:.2f}%)\n'
            f'    MV:           {n_v_lost_mv_1o2} ({f_v_lost_mv_1o2:.2f}%)\n'
            f'    MV1 only:     {n_v_lost_mv1_o} ({f_v_lost_mv1_o:.2f}%)\n'
            f'    MV2 only:     {n_v_lost_mv2_o} ({f_v_lost_mv2_o:.2f}%)\n'
            f'    MV1 and MV2:  {n_v_lost_mv_1a2} ({f_v_lost_mv_1a2:.2f}%)\n'
        )

    if plot:
        # Plot number of values lost when removing features with MV
        df_show = pd.DataFrame({
            'N values lost': [
                n_v_lost_mv1,
                n_v_lost_mv2,
                n_v_lost_mv_1o2,
                # n_v_lost_mv1_o,
                # n_v_lost_mv2_o,
                # n_v_lost_mv_1a2,
            ],
            'N R': [n_rows for _ in range(3)],
            # 'N R': [n_rows for _ in range(6)],
            'type': [
                f'Not applicable\n{f_v_lost_mv1:.2f}%',
                f'Not available\n{f_v_lost_mv2:.2f}%',
                f'Not applicable or\nnot available\n{f_v_lost_mv_1o2:.2f}%',
                # f'MV1 only\n{f_v_lost_mv1_o:.2f}%',
                # f'MV2 only\n{f_v_lost_mv2_o:.2f}%',
                # f'MV1 and MV2\n{f_v_lost_mv_1a2:.2f}%'
            ],
        })

        df_show.sort_values('N values lost', ascending=False, inplace=True)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        sns.set_color_codes('muted')
        sns.barplot(x='N values lost', y='type', data=df_show, color='b', ax=ax)

        box = ax.get_position()
        ax.set_position([1.1*box.x0, box.y0, box.width, box.height])

        ax.set_title('Number of non-missing values lost'
                     '\nwhen removing features containing missing values of type:')
        ax.set(ylabel='', xlabel=f'Number of values (Total {n_values})')
        ax.set_xlim(right=n_values)
        sns.despine(left=True, bottom=True, ax=ax)


def figure1(indicators, plot=True, db_name=None, table=None):
    """Print a global overview of missing values"""
    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 6))
    fig1.tight_layout(pad=2)
    if all((db_name, table)):
        fig1.suptitle(f'Overview of missing values in {db_name} (table "{table}")',
                      fontsize='xx-large')

    matplotlib.rcParams.update({'font.size': 13})

    plot_global(indicators, plot=plot, ax=axes1[0])
    plot_features(indicators, plot=plot, ax=axes1[1])
    plot_rows(indicators, plot=plot, ax=axes1[2])


def figure2(indicators, plot=True, db_name=None, table=None):
    matplotlib.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })
    fig2, _ = plot_feature_wise(indicators, plot=plot)
    if all((db_name, table)):
        fig2.suptitle(f'Proportion of missing values in each feature'
                      f'\nof {db_name} (table "{table}")',
                      fontsize='x-large')


def figure2bis(indicators, plot=True, db_name=None, table=None):
    matplotlib.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })
    fig2, *_ = plot_feature_wise_v2(indicators, plot=plot)
    if all((db_name, table)):
        fig2.suptitle(f'Proportion of missing values in each feature'
                      f'\nof {db_name} (table "$\\verb|{table}|$")',
                      fontsize='x-large')


def figure3(indicators, plot=True, db_name=None, table=None):
    matplotlib.rcParams.update({
        # 'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    fig3, axes3 = plt.subplots(2, 1, figsize=(10, 8))
    fig3.tight_layout(pad=5, h_pad=7, rect=(0.05, 0, 1, .92))
    if all((db_name, table)):
        fig3.suptitle(f'Effect of removing features containing missing values'
                      f'\non {db_name} (table "{table}")',
                      fontsize='x-large')

    plot_rm_rows(indicators, plot=plot, ax=axes3[0])
    plot_rm_features(indicators, plot=plot, ax=axes3[1])


def plot_feature_types(props, ax=None):
    matplotlib.rcParams.update({
        # 'font.size': 14,
        # 'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 6.5))

    props.reset_index(inplace=True)
    props['tag'] = props['tag'].str.replace('_', '-')
    props['task'] = props['task'].str.replace('_', r'\_')
    props['task'] = props['task'].str.replace('pvals', r'screening')

    # Compute cumsums for plotting
    props['categorical+ordinal'] = props['categorical'] + props['ordinal']
    props['continue+ordinal'] = props['continue'] + props['ordinal']

    c1, c2, c3 = sns.color_palette('deep', n_colors=3)
    g1 = sns.barplot(y='tag', x='n', data=props, orient='h', hue='T', color=c1, palette=[c1], ax=ax)
    g2 = sns.barplot(y='tag', x='categorical+ordinal', data=props, orient='h', hue='T', color=c3, palette=[c3], ax=ax)
    g3 = sns.barplot(y='tag', x='categorical', data=props, orient='h', hue='T', color=c2, palette=[c2], ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[handles[10], handles[5], handles[0]],
              labels=['Categorical', 'Ordinal', 'Continue'],
              title='Feature type', fancybox=True, shadow=False,
              loc='upper center', bbox_to_anchor=(0.215, 1.17), ncol=3)
    ax.set_xlabel('Features')
    ax.set_ylabel('')

    props.set_index(['db', 'task', 'T'], inplace=True)
    subprops = props.iloc[props.index.get_level_values('T') == 0]
    minor_YT = ax.get_yaxis().get_majorticklocs()
    subprops['YT_V'] = minor_YT
    major = subprops.groupby(level=[0])['YT_V'].first()
    ax.set_yticklabels(subprops.index.get_level_values(1))

    xlim = ax.get_xlim()

    for i, (idx, value) in enumerate(major.items()):
        ax.text(-40, major.iloc[i], idx, va='center', ha='right')

    ax.set_xlim(xlim)
