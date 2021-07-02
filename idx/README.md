# Indexes of the sampled datasets

## Content
* 1410 files containing indexes of the sub-sampled datasets in the `idx/` folder.
* `idx.md5` containing md5sum values of the files in `idx/`.
* `read.py` a python script to read the files in the `idx/` folder.
* `README.pdf` a pdf version of this file.


Indexes of the sub-sampled datasets for train and test sets are listed in `idx/` folder in
files of the form:
```python
{task}-size{size}-trial{trial}-fold{fold}-train-idx.csv  # Train set indexes
{task}-size{size}-trial{trial}-fold{fold}-test-idx.csv  # Test set indexes
```
where:

* {task} is the name of the task (listed below),
* {size} is one of [2500, 10000, 25000, 100000] (not that all sizes are not
available with all tasks),
* {trial} is one of [1, 2, 3, 4, 5] for tasks ending with "_screening" and
equal to 1 otherwise,
* {fold} is one of [1, 2, 3, 4, 5].


Names of the 13 tasks are:
```python
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
```

Each csv file contains 1 column  with the name of the index column of the
database given as a header on the first line.

## Read indexes files
Files can be read using python as follows:
```python
from os.path import exists
import pandas as pd


# tasks = [...]  # Given above

for task in tasks:
    for size in [2500, 10000, 25000, 100000]:
        for trial in range(1, 6):
            for fold in range(1, 6):
                train_file = f'idx/{task}-size{size}-trial{trial}-fold{fold}-train-idx.csv'
                test_file = f'idx/{task}-size{size}-trial{trial}-fold{fold}-test-idx.csv'

                # Some sizes and trials are not available for some tasks
                if not exists(train_file) or not exists(test_file):
                    continue

                train_idx = pandas.read_csv(train_file, header=0, squeeze=True)
                test_idx = pandas.read_csv(test_file, header=0, squeeze=True)

                # Process the files...

```

## Checksum
md5sum values for each of the above files are listed in `idx.md5`.
To check the integrity of the downloaded files, run:
```
md5sum --check idx.md5
```
