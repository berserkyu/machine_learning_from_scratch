from abc import abstractmethod
from src.algos.constants import params_constraints

class Model:
    def __init__(self, init_params={}):
        self.set_params(init_params)

    def get_params(self):
        return self._params
    
    def set_params(self, params):
        for param, constraint in params_constraints.items():
            if param in params and not constraint(params[param]):
                raise ValueError(f"Invalid value for parameter '{param}': {params[param]}")
        self._params = params

    @abstractmethod
    def fit(self, X, Y, params):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    @abstractmethod
    def explain(self):
        pass

    

