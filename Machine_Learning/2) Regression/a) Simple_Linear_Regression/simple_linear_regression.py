#simple linear regression

#importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("Salary_Data.csv")

#creating matrix of feature
X = dataset.iloc[:,0].values

#creating dependent variable vector
y = dataset.iloc[:,1].values

#splitting dataset into train set and test set
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)
X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)


#Fitting simple linear regression into training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set result
y_pred = regressor.predict(X_test)


#visualizing training set result
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Salary")
plt.ylabel("years of experience")
plt.show()

#visualizing test set result
plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Salary")
plt.ylabel("years of experience")
plt.show()