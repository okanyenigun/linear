import pandas as pd
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
