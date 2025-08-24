### Linear Regression
- Implement a linear regression model that can take N dimensional data.

- The model class should have 3 features:
    1. train
    2. predict 
    3. explain

- The train method should train the model with gradient descent, the method should accept the following params:
    1. Training features X, (m,n) dataframe
    2. Target variable y, 1-d array of length M
    3. learning rate, float
    4. batch_size (of total training sample), float
    5. cost_function, str/enum
    6. gradient descent method, str/enum
    7. num iterations
    8. early stopping steps

- The predict method should output a prediction given data X, the method should accept the followuing params:
    1. Data X, (m, N) dataframe

- and returns:
    1. Prediction Y, 1-d array of length m

- The explain method should geenrate a report/charts & explain what the model is doing, e.g. feature weight, feature intercept, SHAP value, etc, the method should accept params:
    1. Background dataset X, (m,n) dataframe

- and returns:
    1. df_weight_smy: summary of weight & intercept of each feature
    2. df_shap_smy: summary of shapley values by feature
    3. shap_bee_swarm: bee swarm plot of SHAP value

