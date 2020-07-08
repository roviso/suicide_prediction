# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('master.csv', thousands=',')
X = dataset.iloc[:, [0,1,2,3,5,8,9,10]].values
y = dataset.iloc[:, 4:5].values

# taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant',fill_value=0 )
imputer = imputer.fit(X[:,5:6])
X[:,5:6] = imputer.transform(X[:,5:6])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,0] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])


#catagoriging into male and female
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [2])],   
    remainder='passthrough'                         
    )
X = np.array(ct.fit_transform(X), dtype=np.float)

#catagoriging based on generation / age-group
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#slpilting into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#fitting SVR to the database
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

#Visualising the SVR results
plt.scatter(X_train,y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('Suicide prediction using SVR model')
plt.xlabel('independent variable')
plt.ylabel('no of suicide')
plt.show()
