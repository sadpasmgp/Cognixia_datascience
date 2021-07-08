import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt

import os
os.chdir('C:\\Users\\HP\\Desktop\\Python Course - Abhaya\\Module 12 - Time Series Analysis\\Project 2\\')
df = pd.read_csv('airline_passengers.csv')

df.head()
df.tail()

df.columns = ['Month','Passengers']
df.head()

# Weird last value at bottom causing issues
df.drop(144,axis=0,inplace=True)

df['Month'] = pd.to_datetime(df['Month'])

df.head()

df.set_index('Month',inplace=True)

df.head()

df.describe().transpose()

df.plot()


timeseries = df['Passengers']

timeseries.rolling(12).mean().plot(label='12 Month Rolling Mean')
timeseries.rolling(12).std().plot(label='12 Month Rolling Std')
timeseries.plot()
plt.legend()


timeseries.rolling(12).mean().plot(label='12 Month Rolling Mean')
timeseries.plot()
plt.legend()


#ETS
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['Passengers'], freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


"""
## Testing for Stationarity

We can use the Augmented [Dickey-Fuller](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test) [unit root test](https://en.wikipedia.org/wiki/Unit_root_test).

In statistics and econometrics, an augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root is present in a time series sample. The alternative hypothesis is different depending on which version of the test is used, but is usually stationarity or trend-stationarity.

Basically, we are trying to whether to accept the Null Hypothesis **H0** (that the time series has a unit root, indicating it is non-stationary) or reject **H0** and go with the Alternative Hypothesis (that the time series has no unit root and is stationary).

We end up deciding this based on the p-value return.

* A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis.

* A large p-value (> 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis.

Let's run the Augmented Dickey-Fuller test on our data:
"""
df.head()

#To check the data is stationary or not (Null hypothesis non stationary) 

from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Milk in pounds per cow'])

"""
print('Augmented Dickey-Fuller Test:')
labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

for value,label in zip(result,labels):
    print(label+' : '+str(value) )
    
if result[1] <= 0.05:
    print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
else:
    print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    
    
    
# Store in a function for later use!
def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        
     

df['Milk First Difference'] = df['Milk in pounds per cow'] - df['Milk in pounds per cow'].shift(1)




adf_check(df['Milk First Difference'].dropna())



df['Milk First Difference'].plot()


# Sometimes it would be necessary to do a second difference 
# This is just for show, we didn't need to do a second difference in our case
df['Milk Second Difference'] = df['Milk First Difference'] - df['Milk First Difference'].shift(1)


adf_check(df['Milk Second Difference'].dropna())


df['Milk Second Difference'].plot()


df['Seasonal Difference'] = df['Milk in pounds per cow'] - df['Milk in pounds per cow'].shift(12)
df['Seasonal Difference'].plot()




# Seasonal Difference by itself was not enough!
adf_check(df['Seasonal Difference'].dropna())




# You can also do seasonal first difference
df['Seasonal First Difference'] = df['Milk First Difference'] - df['Milk First Difference'].shift(12)
df['Seasonal First Difference'].plot()



adf_check(df['Seasonal First Difference'].dropna())



"""
# For non-seasonal data
from statsmodels.tsa.arima_model import ARIMA



# I recommend you glance over this!

# 
help(ARIMA)


# We have seasonal data!
model = sm.tsa.statespace.SARIMAX(df['Passengers'],order=(0,1,0), seasonal_order=(1,1,1,12))
results = model.fit()
print(results.summary())


results.resid.plot()


results.resid.plot(kind='kde')


df['forecast'] = results.predict(start = 126, end= 144, dynamic= True)  
df[['Passengers','forecast']].plot(figsize=(12,8))


df.tail()


from pandas.tseries.offsets import DateOffset

future_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,24) ]


future_dates

future_dates_df = pd.DataFrame(index=future_dates[1:],columns=df.columns)


future_df = pd.concat([df,future_dates_df])

future_df.head()

future_df.tail()

future_df['forecast'] = results.predict(start = 145, end = 169, dynamic= True)  
future_df[['Passengers', 'forecast']].plot(figsize=(12, 8)) 