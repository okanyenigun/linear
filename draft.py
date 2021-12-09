import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import load_boston
#%% dataset
boston_dataset = load_boston()
df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
df['MEDV'] = boston_dataset.target

#%% OLS
#simple regression
X = df["RM"]
y = df["MEDV"]
simple_model = sm.OLS(y, X).fit()
predictions = simple_model.predict(X)
print(simple_model.summary())
print("***********************")

#add constant to simple model
X = sm.add_constant(X)
simple_constant_model = sm.OLS(y, X).fit()
predictions = simple_constant_model.predict(X)
simple_constant_model.summary()
print(simple_constant_model.summary())
print("***********************")
#multi

X = df.iloc[:,:13].values
multi_model = sm.OLS(y, X).fit()
predictions = multi_model.predict(X)
print(multi_model.summary())
print("***********************")

#%%

X = df["RM"]
y = df["MEDV"]

def mean(values):
    return sum(values) / float(len(values))

def variance(values,mean):
    """The variance is the sum squared difference for each value from the mean value."""
    return sum([(x-mean)**2 for x in values])

def covariance(x,mean_x,mean_y):
    """The covariance of two groups of numbers describes how those numbers change together."""
    covar = 0.0
    for i in range(len(x)):
    	covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar

#%%

def h(x,theta):
    """
    hθ(x) = θ0 + θ1x1 + θ2x2 + θ3x3 +…..+ θnxn
    matmul: matrix multiplication
    """
    return np.matmul(x,theta)
    
def cost_function(x,y,theta):
    """
    
    The @ operator can be used as a shorthand for np.matmul on ndarrays.
    """
    return ((h(x, theta)-y).T@(h(x, theta)-y))/(2*y.shape[0])


def gradient_descent(x,y,theta,learning_rate=0.1,num_epochs=10):
    m = x.shape[0]
    J_all = []
    for _ in range(num_epochs):
        h_x = h(x,theta)
        cost_ = (1/m)*(x.T@(h_x-y))
        theta = theta - (learning_rate)*cost_
        J_all.append(cost_function(x,y,theta))
    
    return theta,J_all

def add_constant(x):
    ones = np.ones((X.shape[0],1))
    return np.concatenate((x,ones),axis=1)

#%%
X = df.iloc[:,:13].values
y = df["MEDV"]
X = add_constant(X)
theta = np.zeros((X.shape[1], 1))
learning_rate = 0.1
num_epochs = 50
theta, J_all = gradient_descent(X, y, theta, learning_rate, num_epochs)
J = cost_function(X, y, theta)
print("Cost: ", J)
print("Parameters: ", theta)

#for testing and plotting cost 
n_epochs = []
jplot = []
count = 0
for i in J_all:
	jplot.append(i[0][0])
	n_epochs.append(count)
	count += 1
jplot = np.array(jplot)
n_epochs = np.array(n_epochs)

