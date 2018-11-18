
import numpy as np 
import pandas as pd 

import os

# Any results you write to the current directory are saved as output.

dataset = pd.read_csv("train_V2.csv")

dataset.drop(['matchType'],axis=1,inplace=True)
pd.DataFrame(dataset).fillna(0, inplace=True)

X_train = dataset.iloc[:1000, 3:27].values  
y_train = dataset.iloc[:1000, 27].values

dataset1 = pd.read_csv("test_V2.csv")

dataset1.drop(['matchType'],axis=1,inplace=True)
pd.DataFrame(dataset1).fillna(0, inplace=True)

ID = dataset1.iloc[:, 0:1].values
X_test = dataset1.iloc[:, 3:].values  


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)

ID = ID.reshape(-1)
fmt='%s, %.8f'
combined = np.vstack((ID, y_pred)).T
 
np.savetxt("submission_V2.csv", combined, header="Id,winPlacePerc", delimiter=",",fmt=fmt,comments='')

