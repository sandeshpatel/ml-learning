
# coding: utf-8

# In[8]:
class linear_model:

    def __init__(self, X, y):
        self.X = X
        self.theta = np.zeros((self.X.shape[1],1))
        self.y = np.atleast_2d(y)
        print(y.shape)



    def fit(self):

        print(self.theta)
        self.grad_desc()


    def grad_desc(self):
        for i in range(7000):
            self.theta = self.theta - 0.1/40*(np.transpose(np.transpose((self.X @ self.theta) - self.y ) @ self.X))
        print(self.theta)


    def predict(self, X):
            return X @ self.theta



# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

#encoding catagorical data state
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoind the dummy variable trap
X = X[:, 1:]

X = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
#y_test = sc_y.fit_transform(y_test)


lin_mod= linear_model(X_train, y_train)
lin_mod.fit()
print(lin_mod.predict(X_test))
print(y_test)
