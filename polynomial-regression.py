#### Global requirements
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#### Reading the data (it's the same as the one in linear-regression.py)
data_raw = pd.read_csv('FuelConsumption.csv')

## Get important fields only
data = data_raw[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
#print(data.head())

## Plot a graph with emission related to engine size
"""
plt.scatter(data.ENGINESIZE, data.CO2EMISSIONS, color='blue')
plt.xlabel('Engine Size')
plt.ylabel('Co2 Emissions')
plt.show()
"""

#### Polynomial regresison model
## Create train and test datasets
mask = np.random.rand(len(data_raw)) < 0.8
train = data_raw[mask]
test = data_raw[~mask]

## Create models
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

## Transform models from a one dimensional array to a 2 dimensional array
poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)

# To see how the data has been transformed you can uncoment these print lines below
#print(train_x)
#print(train_x_poly)

## Make linear regression
linearReg = linear_model.LinearRegression()
train_y = linearReg.fit(train_x_poly, train_y) # Fit engine size in 2 dimensional array with co2 emissions

# To see the coefficients uncoment these print lines below
#print('Coefficients: ', linearReg.coef_)
#print('Intercept: ', linearReg.intercept_)

# To see a graph with these informations, remove the comment bellow
"""plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
dimension_x = np.arange(0.0, 10.0, 0.1)
dimension_y = linearReg.intercept_[0] + linearReg.coef_[0][1]*dimension_x + linearReg.coef_[0][2]*np.power(dimension_x, 2)
plt.plot(dimension_x, dimension_y, color='red')
plt.xlabel('Engine Size')
plt.ylabel('Co2 Emissions')
plt.show()"""

#### Evaluation of the model
from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_evaluate = linearReg.predict(test_x_poly)

# To see the evaluation, uncoment these print lines below
#print('Mean absolute error (the lower the better): {:.2f}'.format( np.mean(np.absolute(test_y_evaluate - test_y)) ))
#print('Residual sum of squares (the lower the better): {:.2f}'.format( np.mean(test_y_evaluate - test_y) ** 2 ))
#print('R2 Score (the higher the better): {:.2f}'.format( r2_score(test_y, test_y_evaluate) )) 


#### Conclusions
"""
# As it was observable, although the graph of the 2-dimensional array looked to fit the data better,
# the simple linear regression had higher R2 scores, both when using only FUELCONSUMPTION_COMB and
# FUELCONSUMPTION_CITY + FUELCONSUMPTION_HWY
"""