from .NHIS import NHIS
from .TB import TB
from .UKBB import UKBB
from .UKBB_utils import UKBB_feature_types_converter
from .MIMIC import MIMIC
from features_type import _load_feature_types

dbs = {
    'TB': TB(),
    'UKBB': UKBB(),
    'MIMIC': MIMIC(),
    'NHIS': NHIS(),
}
