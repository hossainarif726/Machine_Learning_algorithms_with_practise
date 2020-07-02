import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct=ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X = np.array(ct.fit_transform(X),dtype=np.float)

#Avoiding DummyTrap
X = X[:,1:]

#Splitting dataset inti train set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=0)

#Fitting training set into regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting
y_pred = regressor.predict(X_test)


import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()
regressor_OLS.summary()
