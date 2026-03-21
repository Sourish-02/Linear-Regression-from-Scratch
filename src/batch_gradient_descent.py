import numpy as np
import pandas as pd

data= pd.read_csv("train.csv")
X_train = data['x'].values.reshape(-1,1)
y_train = data['y'].values.reshape(-1,1)
m= len(y_train) # Necessary to compute cost function
X_train = np.c_[np.ones(X_train.shape[0]), X_train] # Adding bias

theta = np.zeros((X_train.shape[1], 1)) # A bad guess for theta
alpha= 0.0001
iterations= 1000

def cost_function(X, y, theta):
    y_pred = X @ theta
    cost = (1/(2*m)) * np.sum((y_pred - y)**2)
    return cost

prev_cost = float('inf')
for t in range(iterations):
    gradient= (1/m)*(X_train.T @ (X_train @ theta - y_train))
    theta= theta - (alpha * gradient)
    cost = cost_function(X_train, y_train, theta)
    if abs(prev_cost - cost) < 1e-6:
        break
    prev_cost= cost

check= pd.read_csv("test.csv")
X_test = check['x'].values.reshape(-1,1)
y_test = check['y'].values.reshape(-1,1)
X_test = np.c_[np.ones(X_test.shape[0]), X_test]
y_pred = X_test @ theta

Mean_squared_error = np.mean((y_test - y_pred)**2)
print("MSE:", Mean_squared_error)

# MSE: 9.462345076972634 on taking learning rate= 0.0001
