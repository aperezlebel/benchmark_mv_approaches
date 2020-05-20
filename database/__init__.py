from .NHIS import NHIS
from .TB import TB
from .UKBB import UKBB
from .UKBB_utils import UKBB_feature_types_converter
# from .MIMICIII import MIMICIII
from features_type import _load_feature_types

dbs = {
    'TB': TB(),
    'UKBB': UKBB()
}
