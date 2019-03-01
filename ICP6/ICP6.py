import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#==============
#Q1
#==============

#configure plot
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

#read data and drop null values
train = pd.read_csv('train.csv')
data = train.select_dtypes(include=[np.number]).dropna()

#find outliers (more than 2 STD away from mean) and remove them
data = data[np.abs(data.GarageArea-data.GarageArea.mean()) <= (2*data.GarageArea.std())]

#plot the results
prices = data.SalePrice.get_values()
garageAreas = data.GarageArea.get_values()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(prices,garageAreas)
plt.show()

#==============
#Q2
#==============

#read data and drop or interpolate null values
train = pd.read_csv('weatherHistory.csv')
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

#Find correlated features of a target feature
targetFeature = "Temperature"
numeric_features = train.select_dtypes(include=[np.number])

corr = numeric_features.corr()
print (corr[targetFeature].sort_values(ascending=False)[:5], '\n')
print (corr[targetFeature].sort_values(ascending=False)[-5:])

##Build a linear model for Temperature against Humidity and Visibility
y = data.Temperature
X = data[["Visibility","Wind Bearing (degrees)"]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

##visualize
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Temperature')
plt.ylabel('Actual Temperature')
plt.title('Linear Regression Model')
plt.show()

