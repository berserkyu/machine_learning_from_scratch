import math
import numpy as np

class Loss:
    
    def __init__(self, type='mse'):
        if type not in ['mse','rmse']:
            raise ValueError("Loss type must be one of ['mse','rmse']")
        self._type = type
        pass

    def calculate_loss(self, y, y_hat):
        if self._type == 'mse':
            return self.__calculate_mse(y, y_hat)
        
    def get_gradient(self, weights, intercepts, X, Y):
        if self._type == 'mse':
            return self.__get_gradient_mse(weights, intercepts, X, Y)

    def __calculate_mse(self, y, y_hat):
        delta = np.array(y) - np.array(y_hat)
        delta = delta ** 2

        return delta.mean() 
    
    def __get_gradient_mse(self, weights, intercepts, X, Y):
        N = len(Y)
        M = len(weights)

        y_hat_partial = np.dot(X, weights) + intercepts
        weights_gradient = np.dot(y_hat_partial - Y, X).sum() /N
        intercepts_gradient = (y_hat_partial - Y).sum() / N

        return weights_gradient, intercepts_gradient