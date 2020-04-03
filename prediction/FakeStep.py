import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Print also in console.


class FakeStep:
    def __init__(self, name):
        self.name = name

    def fit(self, X, y):
        logger.info(f'{self.name}: fit called on shape {X.shape}')

        return self

    def transform(self, X):
        logger.info(f'{self.name}: transform called on shape {X.shape}')

        return X
