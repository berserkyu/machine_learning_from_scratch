params_constraints = {
    'learning_rate': lambda x: isinstance(x, (int, float)) and x > 0,
    'batch_size': lambda x: isinstance(x, int) and x > 0,
    'cost_function': lambda x: x in ['rmse', 'mae', 'mse', 'logloss'],
    'gradient_method': lambda x: x in ['stochastic', 'mini-batch', 'batch'],
    'num_iterations': lambda x: isinstance(x, int) and x > 0,
    'early_stopping': lambda x: isinstance(x, bool),
    'early_stopping_steps': lambda x: isinstance(x, int) and x > 0,
    'min_delta': lambda x: isinstance(x, (int, float)) and x >= 0,
    'verbose': lambda x: isinstance(x, bool)
}