import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

dataset = pd.read_csv("Data.csv")

#Matrix of features 
X = dataset.iloc[:,:-1].values

#Dependent variable vector
y = dataset.iloc[:,3].values

#splitting dataset training set and test set 
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)"""