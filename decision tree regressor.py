#Importing the Libraries
 
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt

#Load the dataset
   dataset=pd.read_csv('# your csv file ')

   X= dataset.iloc[].values         # X is a independent variable

   y= dataset.iloc[].values         # y is a dependent variable

#Encoding categorical data into numerical if required
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
  
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features =[3])  
X=onehotencoder.fit_transform(X).toarray()

#Apply feature scaling if required
  from sklearn.preprocessing import StandardScaler
  sc_X=StandardScaler()
  sc_y=StandardScaler()
  X=sc_X.fit_transform(X)
  y=sc_y.fit_transform(y)

#fitting reg model to the dataset
  from sklearn.tree import DecisionTreeRegressor
  regressor=DecisionTreeRegressor(random_state=0)
  regressor.fit(X,y)

#predicting test set result
  y_pred=regressor.predict()

#Visualising the result 
  X_grid=np.arange(min(X),max(X), 0.01)
  X_grid=X_grid.reshape((len(X_grid),1))
  plt.scatter(X,y,color='red')
  plt.plot(X_grid,regressor.predict(X_grid),color='blue')
  plt.title('Decision Tree Regressor')
  plt.xlabel('')
  plt.ylabel('')
  plt.show()