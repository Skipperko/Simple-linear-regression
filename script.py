import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Converting .xls into .csv 
read_file = pd.read_excel ("dataset_c3.xlsx")
read_file.to_csv ("datasetCSV.csv", index = None, header=True)

# Removing the the first column
stripped_file = pd.read_csv('datasetCSV.csv')
stripped_file = stripped_file.drop(['Meno'], axis=1)
stripped_file.to_csv('datasetCSV.csv', index=False)

# Importing the dataset
dataset = pd.read_csv('datasetCSV.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title ('Hours Studied vs Grade (Training set)')
plt.xlabel ('Hours Studied')
plt.ylabel ('Grade')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title ('Hours Studied vs Grade (Test set)')
plt.xlabel ('Hours Studied')
plt.ylabel ('Grade')
plt.show()