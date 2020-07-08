# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:39:33 2020

@author: domin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('master.csv', thousands=',')
X = dataset.iloc[:, [0,1,2,3,5,8,9,10]].values
y = dataset.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant',fill_value=0 )
imputer = imputer.fit(X[:,5:6])
X[:,5:6] = imputer.transform(X[:,5:6])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])￼
X[:,0] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

y_predict = regressor.predict(X_test)

plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('Suicide prediction using SVR model')
plt.xlabel('independent variable')
plt.ylabel('no of suicide')
plt.show()


