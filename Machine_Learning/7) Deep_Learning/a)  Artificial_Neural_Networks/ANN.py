import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn_Modelling.csv")

#Matrix of features 
X = dataset.iloc[:, 3:13].values

#Dependent variable vector
y = dataset.iloc[:, 13].values

# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1:]

#splitting dataset training set and test set 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#creating leyars
import keras
from keras.layers import Dense
from keras.models import Sequential
classifier = Sequential()
classifier.add(Dense(output_dim=6, init='uniform', input_dim=11, activation='relu'))
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)

#Predicting the test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)