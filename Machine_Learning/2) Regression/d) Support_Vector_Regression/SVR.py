#Importing library and dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')

#creating features of matrix and dependent variable vector
X = dataset.iloc[:,1:2].values
X = X.reshape((-1,1))
y = dataset.iloc[:,2].values
y = y.reshape((-1,1))

#splitting dataset training set and test set 
"""from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)"""

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Fitting the regression model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #Radial Basis Function
regressor.fit(X,y)

#Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))


#Visualizing the regression result
plt.scatter(X,y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


#Visualizing the regression result in high resolution
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
