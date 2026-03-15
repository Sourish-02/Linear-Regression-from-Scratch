import numpy as np
import pandas as pd

data= pd.read_csv("train.csv")
X_train = data['x'].values.reshape(-1,1)
y_train = data['y'].values.reshape(-1,1)
X_train = np.c_[np.ones(X_train.shape[0]), X_train] # Column concatenation of bias (a column of all 1)

# theta = np.linalg.inv(X.T @ X) @ X.T @ y
theta = np.linalg.pinv(X_train) @ y_train
# We compute the Moore-Penrose Pseudo-inverse of X, X+= (((X^T)(X))^(-1))(X^T)

check= pd.read_csv("test.csv")
X_test = check['x'].values.reshape(-1,1)
y_test = check['y'].values.reshape(-1,1)
X_test = np.c_[np.ones(X_test.shape[0]), X_test]
y_pred = X_test @ theta

Mean_squared_error = np.mean((y_test - y_pred)**2)
print("MSE:", Mean_squared_error)

Residual_sum_of_squares = np.sum((y_test - y_pred)**2)
Total_sum_of_squares = np.sum((y_test - np.mean(y_test))**2)
r2 = 1 - (Residual_sum_of_squares/Total_sum_of_squares)
print("R2 score:", r2)

# MSE: 9.434852832251472
# R2 score: 0.9887991524196075

import matplotlib.pyplot as plt

plt.scatter(X_test[:,1], y_test, label="Actual data")
plt.plot(X_test[:,1], y_pred, color='red', label="Regression line")
plt.legend()
plt.savefig("regression_plot.png")
plt.show()
