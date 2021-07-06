# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 14:25:20 2018

@author: Hemant Rathore
"""


# 6. Timeseries Analysis using Seasonal ARIMA

## Case Study - Seasonal ARIMA Model for Industrial production of electric and gas utilities 
## in the United States, from the years 1985–2018, with our frequency being Monthly.


import os

os.chdir("D:\Data Science Training\Data")

import pandas as pd

import numpy as np

energy_data = pd.read_csv("IPG_Dataset.csv")



energy_data.head(10)

energy_data.describe()



# Plot the TS & Notice trend,seasonality,heteroscedasticity in TS

energy_data.plot(figsize=(12,12))

# Decompose TS

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(energy_data.PRODUCTION, model='multiplicative',freq=12)

from pylab import rcParams

rcParams['figure.figsize'] = 10,10

result.plot()



## Summary

# Hetro.- yes
# trend- yes
# Seasonality - yes




# 1. Variance is increasing so take log to remove heteroscedasticity

import numpy as np

import pandas as pd

energy = energy_data.PRODUCTION

log_energy = np.log(energy)

log_energy.plot()
energy.plot()




# 2 Detrend TS by differencing 

from pandas import Series

d_log_energy = Series.diff(log_energy,periods=1)

d_log_energy.plot()


# 3 remove Seasonality by taking seasonal difference


dd_log_energy = Series.diff(d_log_energy,periods=12)

dd_log_energy.plot(figsize=(10,5))


## Model Selection


# ARIMA(p,d,q)[non seasonal], (p,d,q)[12][seasonal]


#(p,1,q)(p,1,q)[12]


# Analyze ACF (MA) and PACF(AR) to conclude model



dd_log_energy.isna()

dd_log_energy.dropna(inplace=True)


import statsmodels.api as sm
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot()
fig = sm.graphics.tsa.plot_acf(dd_log_energy, lags=50, ax=ax)


# non seasonal : MA(2) as non seasonal is cutting  off after 2 lags (ignore 0th lag)
# seasonal : MA(2) cutting off after 2 lags

#(p,1,2)(p,1,2)[12]


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
fig = sm.graphics.tsa.plot_pacf(dd_log_energy, lags=50, ax=ax)


# non seasonal : AR(2) as cutting off after 2 lags
# seasonal : AR(0) because its tailing off

#(2,1,2)(0,1,2)[12]



arima_model = sm.tsa.statespace.SARIMAX(energy,
                                            order=(2,1,2),
                                            seasonal_order=(0,1,2,12),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)


arima_model_fit = arima_model.fit()


arima_model_fit.save


pred = arima_model_fit.get_prediction(start = 900, end = 967, dynamic=False)


ax = energy.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='forecast', alpha=.7)

plt.legend()



ax = energy[850:1000].plot(label='observed',figsize=(15,10),color='red')
pred.predicted_mean.plot(ax=ax, label='forecast', alpha=.7,color='blue')

plt.legend()