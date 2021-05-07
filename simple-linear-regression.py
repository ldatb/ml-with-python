### Requirements ###
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### Reading data ###
# This model consists of 13 colums:
# Model year, Maker, Mode, Class, Engine Size, Cylinders, Transmission, Fuel Type, 
# Fuel Consuption at City, Fuel Consuption at Highway, Fuel Consuption at Combined and Co2 emissions
raw_data = pd.read_csv('FuelConsumption.csv')

# Define important columns
valuable_data = raw_data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

### Plot a graph of each feature vs. Co2 Emissions ###
## Uncoment the plt.show() to see the graph you want
"""
# Fuel Consumption
plt.scatter(valuable_data.FUELCONSUMPTION_COMB, valuable_data.CO2EMISSIONS, color='blue')
plt.xlabel('FUEL CONSUMPTION COMBINED')
plt.ylabel('CO2 EMISSIONS')
#plt.show()

# Engine Size
plt.scatter(valuable_data.ENGINESIZE, valuable_data.CO2EMISSIONS, color='red')
plt.xlabel('ENGINE SIZE')
plt.ylabel('CO2 EMISSIONS')
#plt.show()

# Cylinders
plt.scatter(valuable_data.CYLINDERS, valuable_data.CO2EMISSIONS, color='green')
plt.xlabel('CYLINDERS')
plt.ylabel('CO2 EMISSIONS')
#plt.show()
"""

### Creating train and test datasets ###
# Create models
mask = np.random.rand(len(raw_data)) < 0.8
train = valuable_data[mask]
test = valuable_data[~mask]

# Train models
from sklearn import linear_model as lm
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
lr = lm.LinearRegression()
trainedModels = lr.fit(train_x, train_y)
print('Coefficient: ', trainedModels.coef_)
print('Intercept: ', trainedModels.intercept_)

# Plot outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, lr.coef_[0][0]*train_x + lr.intercept_[0], color='red')
plt.xlabel('Engine Size')
plt.ylabel('Emissions')
#plt.show() # Uncoment this line to show the graph

# Evaluate the model
from sklearn.metrics import r2_score
train_x = np.asanyarray(test[['ENGINESIZE']])
train_y = np.asanyarray(test[['CO2EMISSIONS']])
train_y_eval = lr.predict(train_x)
print('Mean absolute error: {:.2f}'.format(np.mean(np.absolute(train_y_eval - train_y))))
print('Residual number of squares: {:.2f}'.format(np.mean(np.absolute(train_y_eval - train_y) ** 2)))
print('R2 Score: {:.2f}'.format(r2_score(train_y, train_y_eval)))