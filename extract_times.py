import pandas as pd

df = pd.read_csv('history_mi2500.csv', index_col='JobID', delimiter=';',  # dtype={'Start': 'datetime64', 'Stop': 'datetime64'},)
                 parse_dates=['Start', 'End', 'Elapsed', 'ElapsedRaw'])
df = df.query('State == "COMPLETED"')
df = df[df['JobName'].str.contains('^20|^24|^22|^26|^M')]

r = df['JobName'].str.extract(r'^M?(20|24|22|26)T?(.)_?(TB|UKBB|MIMIC|NHIS|T|U|M|N)\/?(.*)')
r.rename({0: 'method', 1: 'trial', 2: 'db', 3: 'task'}, axis=1, inplace=True)

df = pd.concat([df, r], axis=1)

time_col = 'CPUTime'
# time_col = 'CPUTimeRAW'
# time_col = 'Elapsed'
# time_col = 'SystemCPU'
# time_col = 'TotalCPU'
# time_col = 'UserCPU'

def add_hours(s):
    if '.' in s:
        return f'00:{s}'
    return s

df[time_col] = df[time_col].apply(add_hours)
df[time_col] = pd.to_timedelta(df[time_col].str.replace('-', ' days '))
df[time_col] = df[time_col].dt.total_seconds()

df.set_index(['JobName', 'db', 'task', 'method', 'trial', time_col], append=True, inplace=True)
df.rename({'T': 'TB', 'U': 'UKBB', 'N': 'NHIS', 'M': 'MIMIC'}, axis=0, level='db', inplace=True)

df.sort_index(axis=0, level=['db', 'task', 'method', 'trial', time_col], inplace=True)
df.reset_index(inplace=True)
df.drop_duplicates(subset=['db', 'task', 'method', 'trial'],
                         keep='last', inplace=True)
df.set_index(['db', 'task', 'method', 'trial'], inplace=True)

df = df[[time_col]]

df.to_csv('jobs.csv')

df.reset_index(inplace=True)
df.set_index(['db', 'task', 'method', 'trial'], inplace=True)

scores = pd.read_csv('scores/scores_mi_2500.csv', index_col=0)
scores.set_index(['size', 'db', 'task', 'method', 'trial', 'fold'], inplace=True)

reg_tasks = [
    "TB/platelet_pvals",
    "UKBB/fluid_pvals",
    "NHIS/income_pvals",
]

for index, subscores in scores.reset_index().groupby(['size', 'db', 'task', 'method', 'trial']):
    size, db, task, method, trial = index
    if method == 'MI':
        m_clf = 20
        m_reg = 22
    elif method == 'MI+mask':
        m_clf = 24
        m_reg = 26

    if f'{db}/{task}' in reg_tasks:
        m = m_reg
    else:
        m = m_clf

    time = df.loc[(db, task, str(m), str(trial)), time_col]

    n_folds = len(subscores)
    for fold in subscores['fold']:
        scores.loc[(size, db, task, method, trial, fold), 'tuning_PT'] = time/n_folds

scores.reset_index(inplace=True)
scores.to_csv('scores/scores_mi_2500.csv')

print(scores)
