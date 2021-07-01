import main


tasks = [
    'TB/death_pvals',
    'TB/platelet_pvals',
    'TB/hemo',
    'TB/hemo_pvals',
    'TB/acid',
    'TB/septic_pvals',
    'UKBB/breast_25',
    'UKBB/breast_pvals',
    'UKBB/skin_pvals',
    'UKBB/parkinson_pvals',
    'UKBB/fluid_pvals',
    'MIMIC/septic_pvals',
    'MIMIC/hemo_pvals',
    'NHIS/income_pvals',
]

for task in tasks:
    for T in [0]:
        argv = [
            'run.py',
            'prediction',
            task,
            '0',
            '--T',
            T,
            '--RS',
            '0',
            '--idx',
        ]

        main.run(argv)
