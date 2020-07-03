import logging
import time

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.


class TimerStep:
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
        logger.info(f'{self.name}: fit called on shape {X.shape}')
        self.fit_timestamps.append(time.time())
        self.fit_pts.append(time.process_time())

        return self

    def transform(self, X):
        logger.info(f'{self.name}: transform called on shape {X.shape}')
        self.transform_timestamps.append(time.time())
        self.transform_pts.append(time.process_time())

        return X
