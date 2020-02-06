"""Implement the Strategy class."""


class Strategy():

    def __init__(self, estimator, split, cv, param_space,
                 search):

        self.estimator = estimator
        self.split = split
        self.cv = cv

        if not all(p in estimator.get_params().keys() for p in param_space.keys()):
            raise ValueError('Given parmameters must all be params of estimator.')
        self.param_space = param_space
        self._search = search


    @property
    def search(self):
        return self._search(self.estimator, self.param_space, self.cv)
