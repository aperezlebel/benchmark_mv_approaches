import prediction
import test_fit_time
import statistics
import selection
import extraction


def run(argv=None):
    if argv is None or argv[1] == 'prediction':
        prediction.run(argv[1:])
    elif argv[1] == 'select':
        selection.run(argv[1:])
    elif argv[1] == 'extract':
        extraction.run(argv[1:])
    elif argv[1] == 'extract_UKBB_v1':
        extraction.run_UKBB_v1(argv[1:])
    elif argv[1] == 'extract_UKBB_v2':
        extraction.run_UKBB_v2(argv[1:])
    elif argv[1] == 'test_fit_time':
        test_fit_time.run(argv[1:])
    elif argv[1] == 'stats':
        statistics.run(argv[1:])
    else:
        raise ValueError(f'Unrecognized argument {argv[1]}')
