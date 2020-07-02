#Importing library and dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

#creating features of matrix and dependent variable vector
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Fitting the linear regression model

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting the polynomial regression model to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualizing the linear regression result
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff (Linera Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the polynomia regression result
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color = 'blue')
plt.title('Truth or Bluff (Linera Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#predicting result for linear regression model
lin_reg.predict([[6.5]])

##predicting result for polynomial regression model
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))