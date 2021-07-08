import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'/home/gbhure/Data/Salary_Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=3)
regressor = LinearRegression()
regressor.fit(train_X, train_y)
y_pred = regressor.predict(test_X)

pickle.dump(regressor, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

new_data = np.array([[2.5], [7.9], [17], [19]])
new_pred = model.predict(new_data)
print(new_pred)
