"""Implement the Transform class."""
from dataclasses import dataclass
from typing import Callable, List
import pandas as pd


@dataclass(frozen=True)
class Transform(object):
    """Store a transformer on a dataframe."""

    transform: Callable[[pd.DataFrame], pd.dataFrame] = lambda x: x
    input_features: List[str]
    output_features: List[str]
