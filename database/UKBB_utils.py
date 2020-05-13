import pandas as pd
import os
import numpy as np

from .constants import CONTINUE_R, CONTINUE_I, BINARY, CATEGORICAL, \
    NOT_A_FEATURE, DATE_EXPLODED
from .UKBB import UKBB

from features_type import _dump_feature_types


def UKBB_feature_types_converter(html_folder):
    # Get all files from the html folder
    if not os.path.exists(html_folder):
        raise ValueError(f'Folder not found. {html_folder}')
    (_, _, filenames) = next(os.walk(html_folder))

    for filename in filenames:
        basename, extension = os.path.splitext(filename)

        if extension == '.html':
            html_file = os.path.join(html_folder, filename)

            types = _html_to_types(html_file)
            _dump_feature_types(types, UKBB(), basename, anonymize=False)
            print(types)


def _html_to_types(html_file):
    table = pd.read_html(html_file, match='UDI', header=0)[0]
    table['Type'] = table['Type'].ffill()

    table.set_index('UDI', inplace=True, verify_integrity=True)
    types = table['Type']
    types[types == 'Categorical (multiple)'] = CATEGORICAL
    types[types == 'Categorical (single)'] = CATEGORICAL
    types[types == 'Integer'] = CONTINUE_I
    types[types == 'Continuous'] = CONTINUE_R
    types[types == 'Time'] = DATE_EXPLODED
    types[types == 'Date'] = DATE_EXPLODED
    types[types == 'Sequence'] = NOT_A_FEATURE
    types[types == 'Curve'] = NOT_A_FEATURE
    types[types == 'Text'] = NOT_A_FEATURE

    return types
