"""Implement the TimerStep class."""
import time


class TimerStep:
    """Record timestamps as part of a scikit-learn Pipeline step."""
    def __init__(self, name):
        self.name = name
        self.fit_timestamps = []
        self.transform_timestamps = []
        self.fit_pts = []
        self.transform_pts = []

    @property
    def last_fit_timestamp(self):
        if self.fit_timestamps:
            return self.fit_timestamps[-1]

    @property
    def last_transform_timestamp(self):
        if self.transform_timestamps:
            return self.transform_timestamps[-1]

    @property
    def last_fit_pt(self):
        if self.fit_pts:
            return self.fit_pts[-1]

    @property
    def last_transform_pt(self):
        if self.transform_pts:
            return self.transform_pts[-1]

    def fit(self, X, y):
        self.fit_timestamps.append(time.time())
        self.fit_pts.append(time.process_time())

        return self

    def transform(self, X):
        self.transform_timestamps.append(time.time())
        self.transform_pts.append(time.process_time())

        return X
