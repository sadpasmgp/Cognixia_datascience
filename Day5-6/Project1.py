#Import relevant packages
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

#Setting variables 
file_path = 'D:\\Trainings\\R and Python Classes\\Machine Learning A-Z\\Part 2 - Regression\\Project 1 - Predicting House Price\\'
file_name = 'train.csv'

#Import Data from csv file
df = pd.read_csv(file_path + file_name)


#Four colums to select as per business requirement
dff=df[['LotFrontage','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','SalePrice']]

#Data analysis
dff.head()
dff.isnull().sum()
dff.fillna(dff.median(), inplace=True)

sns.pairplot(dff)

dff.LotFrontage.min()
dff.LotFrontage.max()
dff.LotFrontage.mean()

sns.heatmap(dff.corr(),annot=True)

m1 = sns.distplot(dff.LotFrontage, bins=20, color='r')

#Split the data into dependent and independent variables. 
X = dff.iloc[:, 0:4].values
y = dff.iloc[:, 4].values

#Split the data into Test and Train 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Multi Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)



# Explained variance score: 1 is perfect prediction
r_square = r2_score(y_test, y_pred)
print('Variance score: {0:.2f}'.format(r_square))



#Actual vs Predicted difference 
actual_data = np.array(y_test)
for i in range(len(y_pred)):
    actual = actual_data[i]
    predicted = y_pred[i]
    explained = ((actual_data[i] - y_pred[i])/actual_data[i])*100
    print('Actual value ${:,.2f}, Predicted value ${:,.2f} (%{:.2f})'.format(actual, predicted, explained))
        
        
