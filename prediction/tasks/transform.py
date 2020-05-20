"""Implement the Transform class."""
from dataclasses import dataclass, field
from typing import Callable, List
import pandas as pd


@dataclass(frozen=True)
class Transform(object):
    """Store a transformer on a dataframe."""

    transform: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x
    input_features: List[str] = field(default_factory=list)
    output_features: List[str] = field(default_factory=list)
    child_sep: str = '_'

    def get_infos(self):
        """Return a dict containing infos on the object."""
        return {
            'input_features': self.input_features,
            'output_features': self.output_features,
        }

    def get_parent(self, features):
        """From a set of features, derive the parent features."""
        return {f.split(self.child_sep)[0] for f in features}
