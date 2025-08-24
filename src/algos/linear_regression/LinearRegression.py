import numpy as np
import math
import pandas as pd
from src.utils.loss import Loss


class LinearRegression:  

    _cost_function_options = ['rmse', 'mae', 'mse']
    _gradient_method_options = ['stochastic', 'mini-batch', 'batch']

    def __init__(self, learning_rate=0.01, batch_size=32, cost_function='mse', gradient_method='batch', num_iterations=1000, 
                 early_stopping=True, early_stopping_steps=10):
        self.__set_params(learning_rate, batch_size, cost_function, gradient_method, num_iterations, early_stopping, 
                          early_stopping_steps, verbose=True)
        self._loss_calculator = Loss('mse')
        return
    
    def __set_params(self, learning_rate=None, batch_size=None, cost_function=None, gradient_method=None, num_iterations=None, 
                     early_stopping=None, early_stopping_steps=None, verbose=True):
              
        if learning_rate is not None and learning_rate > 0:
            self._learning_rate = learning_rate
        elif learning_rate is not None:
            raise ValueError("Learning rate must be a positive number.")
        
        if batch_size is not None and batch_size > 0 and isinstance(batch_size, int):
            self._batch_size = batch_size
        elif batch_size is not None:
            raise ValueError("Batch size must be a positive integer.")
        
        if cost_function is not None and cost_function in self._cost_function_options:
            self._cost_function = cost_function
        elif cost_function is not None:
            raise ValueError(f"Cost function must be one of {self._cost_function_options.__str__}.")
        
        if gradient_method is not None and gradient_method in self._gradient_method_options:
            self._gradient_method = gradient_method
        elif gradient_method is not None:
            raise ValueError(f"Gradient method must be one of {self._gradient_method_options.__str__}.")
        
        if num_iterations is not None and num_iterations > 0 and isinstance(num_iterations, int):
            self._num_iterations = num_iterations
        elif num_iterations is not None:
            raise ValueError("Number of iterations must be a positive integer.")
        
        if early_stopping is not None and isinstance(early_stopping, bool):
            self._early_stopping = early_stopping
        else:
            raise ValueError("Early Stopping must be a boolean value.")

        if early_stopping_steps is not None and early_stopping_steps > 0 and isinstance(early_stopping_steps, int):
            self._early_stopping_steps = early_stopping_steps
        elif early_stopping_steps is not None:
            raise ValueError("Early Stopping steps must be a positive integer.") 
        
        if verbose is not None and isinstance(verbose, bool):
            self._verbose = verbose
        else:
            raise ValueError("Verbose must be a boolean value.")

    def fit(self, X, Y, params={}):
        M = X.shape[1]  # number of features
        # initialize param
        if params:
            self.__set_params(learning_rate=params.get('learning_rate', None),
                                batch_size=params.get('batch_size', None),    
                                cost_function=params.get('cost_function', None),
                                gradient_method=params.get('gradient_method', None),
                                num_iterations=params.get('num_iterations', None),
                                early_stopping=params.get('early_stopping', None),
                                early_stopping_steps=params.get('early_stopping_steps', None),
                                verbose=params.get('verbose', None)
                            )
        # random init weight & intercepts
        self._weights = np.random.rand(M).tolist()
        self._intercept = np.random.rand() 
        Y_hat = self.predict(X)

        # prepare for gradient descent
        min_loss = self._loss_calculator.calculate_loss(Y, Y_hat) 
        loss_history = []
        no_improvement_count = 0

        # for num_iteration loops  
        for i in range(self._num_iterations):
            # calculate gradient for every features' wweight & intercept

            weights_gradient, intercepts_gradient = self._loss_calculator.get_gradient(self._weights, self._intercept, X, Y)
            
            # update parameter value
            self._weights = [ w - self._learning_rate * weights_gradient for w in self._weights ]
            self._intercept -= self._learning_rate * intercepts_gradient 
            
            # calculate loss
            Y_hat = self.predict(X)
            loss = self._loss_calculator.calculate_loss(Y, Y_hat)  # depending on cost function
            
            no_improvement_count = no_improvement_count + 1 if loss >= min_loss - 1 and self._early_stopping else 0 # 1 as non-significant improvement

            if self._verbose:
                print(f"Iteration {i+1}: \nweights gradient {weights_gradient}, \nintercepts gradient {intercepts_gradient}")
                print(f"weights {self._weights}, \nintercepts {self._intercept}")
                print(f"loss: {loss}\n")
                print(f"no improvement count: {no_improvement_count}\n")

            # update latest value if its improved
            min_loss = min(min_loss, loss)
            loss_history.append(loss)
            # early stopping if improvement is small 
           
            

            if self._early_stopping_steps and no_improvement_count >= self._early_stopping_steps:
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

    def show_params():
        return {
            "learning_rate": self._learning_rate,
            "batch_size": self._batch_size,
            "cost_function": self._cost_function,
            "gradient_method": self._gradient_method,
            "num_iterations": self._num_iterations,
            "early_stopping": self._early_stopping,
            "early_stopping_steps": self._early_stopping_steps,
            "verbose": self._verbose
        }