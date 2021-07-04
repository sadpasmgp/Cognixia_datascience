# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 14:25:20 2018

@author: Hemant Rathore
"""

## 5  Clustering - K Means Clustering 

## Case Study

## One of Online Cab service providing company has collected trip related data for all
## the drivers working for it, they want to utilize this data to come up with some 
## grouping of the drivers based on the distance and speeding features so that some sort 
## of automation can be done to assign the incoming cab requests to the most appropriate 
## group of drivers

## The main objective is to come up with the clustering solution using given data which can
## effectively divide the drivers into different groups.

## data : Cab Driver data

## Dependent variable - no DV, need to find Drivers Clusters
## Independent variables - Distance travelled, Speed


import os


os.chdir("D:\Data Science Training\Data")

import pandas as pd

import numpy as np

data = pd.read_csv("Driver_data.csv")


data.info()

data.describe()


## Anayzing the distribution of data points by Speed and Distance


import matplotlib.pyplot as plt


fig,axes = plt.subplots(figsize=(12,12))
axes.scatter(data.Distance_Feature,data.Speeding_Feature,alpha=0.9,s=100)




## Feature Scaling


from sklearn import preprocessing


data.drop(['Driver_ID'],axis=1,inplace=True)


data_scaled = preprocessing.scale(data)



fig,axes = plt.subplots(figsize=(12,12))
axes.scatter(data_scaled[:,0],data_scaled[:,1],alpha=0.9,s=100)



## Data Split - not applicable in case of unsupervised learning



## Build K Means Clustering Model

## Step 1 - Finding the value of K using Elbow point graph

## Step 2 - Get the Clustersing solution for given value of K



## Step 1 -

## Calculate "Within cluster sum of squares" for all K values


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, max_iter=1000).fit(data_scaled)



kmeans.inertia_



Within_cluster_sos = {}


for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data_scaled)
    Within_cluster_sos[k] = kmeans.inertia_



Within_cluster_sos.values()


## Drawing Elbow curve - look for the elbow junction point

## Optimum K value = Elbow junction point
    

fig,axes = plt.subplots(figsize=(10,10))

axes.plot(range(1, 11),Within_cluster_sos.values(),'go-')



## Building Final K Means Model


kmeans = KMeans(n_clusters=4, max_iter=1000,init='k-means++').fit(data_scaled)


kmeans.labels_


data['cluster'] = kmeans.labels_



## Analyzing the clusters created by model

import matplotlib.cm as cm
colors = cm.rainbow(data.cluster*100)

fig,axes = plt.subplots(figsize=(12,12))
axes.scatter(data.Distance_Feature,data.Speeding_Feature,c=colors,alpha=0.9,s=100)











# 6. Timeseries Analysis using Seasonal ARIMA

## Case Study - Seasonal ARIMA Model for Industrial production of electric and gas utilities 
## in the United States, from the years 1985–2018, with our frequency being Monthly.


import os

os.chdir("D:\Data Science Training\Data")

import pandas as pd


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
                                            seasonal_order=(0,1,2,12))



arima_model_fit = arima_model.fit()

arima_model_fit.aic

pred = arima_model_fit.get_prediction(start = 900, end = 967, dynamic=False)


ax = energy.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='forecast', alpha=.7)

plt.legend()



ax = energy[850:1000].plot(label='observed',figsize=(15,10),color='red')
pred.predicted_mean.plot(ax=ax, label='forecast', alpha=.7,color='blue')

plt.legend()


## pip install pyramid-arima

from pyramid.arima import auto_arima


stepwise_model = auto_arima(energy, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0,start_Q=0,seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

stepwise_model

print(stepwise_model.aic())