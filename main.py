import prediction
import test_fit_time


def run(argv=None):
    if argv is None or argv[1] == 'prediction':
        prediction.run(argv[1:])
    elif argv[1] == 'test_fit_time':
        test_fit_time.run(argv[1:])
    else:
        raise ValueError(f'Unrecognized argument {argv[1]}')
