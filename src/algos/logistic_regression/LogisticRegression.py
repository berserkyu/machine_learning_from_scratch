from utils.loss import Loss

class LogisticRegression:
    def __init__(self, learning_rate=0.01, batch_size=32, cost_function='mse', gradient_method='batch', num_iterations=1000, 
                 early_stopping=True, early_stopping_steps=10):
        self.__set_params(learning_rate, batch_size, cost_function, gradient_method, num_iterations, early_stopping, 
                          early_stopping_steps, verbose=True)
        self._loss_calculator = Loss('logloss')
    
    def fit(self, X, y):
        pass 