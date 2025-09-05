import numpy as np
import math
import pandas as pd
from src.utils.loss import Loss
from model import Model

class LinearRegression(Model):  

    _cost_function_options = ['rmse', 'mae', 'mse']
    _gradient_method_options = ['stochastic', 'mini-batch', 'batch']
    _default_params = {
        "learning_rate": 0.01,
        "batch_size": 32,
        "cost_function": 'mse',
        "gradient_method": 'batch',
        "num_iterations": 1000,
        "early_stopping": True,
        "early_stopping_steps": 10,
        "min_delta": 0,
        "verbose": False
    }

    def __init__(self, params={}):
        super().__init__()
        self.__set_params(params)
        self._loss_calculator = Loss('mse')
        return

    def fit(self, X, Y, params={}):
        ### Preparation ###
        M = X.shape[1]  # number of features
        if params:
            self.__set_params(params)
        # random init weight & intercepts
        self._weights = np.random.rand(M).tolist()
        self._intercept = np.random.rand() 
        Y_hat = self.predict(X)

        min_loss = self._loss_calculator.calculate_loss(Y, Y_hat) 
        loss_history = []
        no_improvement_count = 0

        ### Iterations ###
        for i in range(self._params["num_iterations"]):
            # calculate gradient for every features' wweight & intercept
            weights_gradient, intercepts_gradient = self._loss_calculator.get_gradient(self._weights, self._intercept, X, Y)
            
            # update parameter value
            self._weights = [ w - self._params["learning_rate"] * weights_gradient for w in self._weights ]
            self._intercept -= self._params["learning_rate"] * intercepts_gradient 
            
            # calculate loss
            Y_hat = self.predict(X)
            loss = self._loss_calculator.calculate_loss(Y, Y_hat)  
            
            # verbose log
            if self._params["verbose"]:
                print(f"Iteration {i+1}: \nweights gradient {weights_gradient}, \nintercepts gradient {intercepts_gradient}")
                print(f"weights {self._weights}, \nintercepts {self._intercept}")
                print(f"loss: {loss}\n")
                print(f"no improvement count: {no_improvement_count}\n")

            # update latest value if its improved
            min_loss = min(min_loss, loss)
            loss_history.append(loss)

            # early stopping check
            no_improvement_count = no_improvement_count + 1 if loss + self._params['min_delta'] >= min_loss and self._params["early_stopping"] else 0 # 1 as non-significant improvement
            if self._params["early_stopping"] and no_improvement_count >= self._params["early_stopping_steps"]:
                print(f"Early stopping at iteration {i+1}")
                break
        print(f"no improvement count: {no_improvement_count}\n")

        return self._weights, self._intercept, loss_history
    
    def predict(self, X: pd.DataFrame):
        results_by_dimension = X.dot(self._weights) + self._intercept
        return results_by_dimension
    
    def explain(self):
        return
    
    def validate(self, X_val, Y_val):
        return