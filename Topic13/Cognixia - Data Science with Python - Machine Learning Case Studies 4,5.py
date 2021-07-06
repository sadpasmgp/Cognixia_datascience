# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 14:25:20 2018

@author: Hemant Rathore
"""

## 4 Random Forest using Regression Tree

## Case Study -
## Data : Wine Quality data

## One of the wine production company is looking for the Rating solution to rate 
## different varieties of its white wines on the scale of 3 to 9, where 3 is the 
## poor quality and 9 is the most superior quality of the wine, they want to 
## consider all the available properties of the wines to make this decision
## The main objective is to find the significant attributes and use their relation
## with wine quality to come up with wine Ratings.

## Dependent variable - Wine quality (3-9)
## Independent variables - All other


import os

os.chdir("D:\Data Science Training\Data")

import pandas as pd

import numpy as np

data = pd.read_csv("winequality-white.csv",sep=';')


data.info()

data.describe()




## Missing data Handling


pd.isna(data.quality)

pd.isna(data.quality).value_counts()




#### 1.2  Categorical Data Handling 

data.info()


object_cols = list(data.select_dtypes(include=['category','object']))

object_cols

data.quality.value_counts()



## 1.3 Feature Scaling


from sklearn import preprocessing

import pandas as pd

data_scaled = preprocessing.scale(data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]])

data1 = pd.DataFrame(data_scaled).join(data['quality'])

data1.columns = data.columns

data = data1


#### 1.4 Splitting data into Training and Test Data Sets

 

from sklearn.model_selection import train_test_split


data.info()

data_X = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]]
data_y = data[:]['quality']


X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=100)

X_train.info()

y_train

X_test.info()

y_test

import seaborn as sns

sns.distplot(data.quality,kde=False)

sns.distplot(y_train,kde=False)

sns.distplot(y_test,kde=False)




## 1.5 Feature Selection -


cr = data.corr()

cr['quality']

sns.heatmap(cr,annot=True,cmap="coolwarm")


## Conclusion - Selected independent variables :  All

## building Classification Model


## 1.6 Build the Model


from sklearn.ensemble import RandomForestRegressor


rf_clf = RandomForestRegressor(n_estimators=5, max_features='auto', random_state=0)


# Train the model using the training sets

rf_clf.fit(X_train,y_train)



## Predicting death_rate for test dataset using model

y_pred = rf_clf.predict(X_test)

y_pred=np.round(y_pred)



res = pd.DataFrame({'y_act':y_test,'y_pred':y_pred})





## 1.7 Model Evaluation - Confusion Matrix

from sklearn.metrics import confusion_matrix


cnf_matrix = confusion_matrix(y_test, y_pred)


print(cnf_matrix)


## RMSE


import math

math.sqrt(((y_test-y_pred)**2).mean())





## Analyze the errors visually



sns.distplot(y_test,kde=False,

             hist_kws={"width":0.5,"alpha": 0.8, "color": "r"}

                       )

sns.distplot(y_pred,kde=False,

             hist_kws={"width":0.5,"alpha": 0.8, "color": "g"}

                       )



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

data_scaled.shape

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
