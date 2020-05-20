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

    def get_infos(self):
        """Return a dict containing infos on the object."""
        return {
            'input_features': self.input_features,
            'output_features': self.output_features,
        }
