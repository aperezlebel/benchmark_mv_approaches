from os.path import exists

import pandas

tasks = [
    'TB-death_screening',
    'TB-platelet_screening',
    'TB-hemo',
    'TB-hemo_screening',
    'TB-acid',
    'TB-septic_screening',
    'UKBB-breast_25',
    'UKBB-breast_screening',
    'UKBB-skin_screening',
    'UKBB-parkinson_screening',
    'UKBB-fluid_screening',
    'MIMIC-septic_screening',
    'MIMIC-hemo_screening',
    'NHIS-income_screening',
]

for task in tasks:
    for size in [2500, 10000, 25000, 100000]:
        for trial in range(1, 6):

            for fold in range(1, 6):
                train_file = f'idx/{task}-size{size}-trial{trial}-fold{fold}-train-idx.csv'
                test_file = f'idx/{task}-size{size}-trial{trial}-fold{fold}-test-idx.csv'

                if not exists(train_file) or not exists(test_file):
                    continue

                train_idx = pandas.read_csv(train_file, header=0, squeeze=True)
                test_idx = pandas.read_csv(test_file, header=0, squeeze=True)

                # Process the files...
                print(train_idx)
                print(test_idx)
