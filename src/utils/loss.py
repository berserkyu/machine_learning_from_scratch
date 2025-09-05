import math
import numpy as np
from src.utils.transformations import sigmoid

class Loss:
    
    def __init__(self, type='mse'):
        if type not in ['mse','rmse','logloss']:
            raise ValueError("Loss type must be one of ['mse','rmse']")
        self._type = type
        pass

    def calculate_loss(self, y, y_hat):
        if self._type == 'mse':
            return self.__calculate_mse(y, y_hat)
        if self._type == 'logloss':
            return self.__calculate_logloss(y, y_hat)
        
    def get_gradient(self, weights, intercept, X, Y):
        if self._type == 'mse':
            return self.__get_gradient_mse(weights, intercept, X, Y)
        if self._type == 'logloss':
            return self.__get_gradient_logloss(weights, intercept, X, Y) 

    def __calculate_mse(self, y, y_hat):
        delta = np.array(y) - np.array(y_hat)
        delta = delta ** 2

        return delta.mean() 
    
    def __calculate_logloss(self, y, y_hat):
        return np.dot(-y , np.log(y_hat)) - np.dot((1-y), (np.log(1 - y_hat)))
    
    def __get_gradient_mse(self, weights, intercept, X, Y):
        N = len(Y)
        M = len(weights)

        y_hat_partial = np.dot(X, weights) + intercept
        weights_gradient = np.dot(y_hat_partial - Y, X).sum() /N
        intercepts_gradient = (y_hat_partial - Y).sum() / N

        return weights_gradient, intercepts_gradient
    
    def __get_gradient_logloss(self, weights, intercept, X, Y):
        N = len(Y)
        M = len(weights)

        y_hat_partial = sigmoid(np.dot(X, weights) + intercept)
        weights_gradient = np.dot(y_hat_partial - Y, X).sum() /N
        intercepts_gradient = (y_hat_partial - Y).sum() / N

        return weights_gradient, intercepts_gradient