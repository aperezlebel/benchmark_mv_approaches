from .NHIS.filter import filter as filter_NHIS
from .UKBB.filter import filter as filter_UKBB
from .TB.filter import filter as filter_TB


def filter(args):
    filter_TB()
    filter_NHIS()
    filter_UKBB()
