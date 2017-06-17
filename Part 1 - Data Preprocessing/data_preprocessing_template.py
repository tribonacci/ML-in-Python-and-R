# Data Preprocessing Template

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset

dataset=pd.read_csv('/Users/v0s003l/Downloads/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 3].values

#Fill missing data
"""from sklearn.preprocessing import Imputer
imp =Imputer(missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True)
imp = imp.fit(X[:, 1:3])
X[:, 1:3] =imp.transform(X[:, 1:3])"""

#Lable encoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_X = LabelEncoder()
X[:, 0]=le_X.fit_transform(X[:, 0])
onx =  OneHotEncoder
one_hot_en = OneHotEncoder(n_values="auto", categorical_features=[0], dtype=np.float64, sparse=False, handle_unknown='error')
X=one_hot_en.fit_transform(X).toarray()
le_Y = LabelEncoder()
Y=le_Y.fit_transform(Y)

#training Test set
from sklearn.cross_validation import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.3,random_state = 55)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

