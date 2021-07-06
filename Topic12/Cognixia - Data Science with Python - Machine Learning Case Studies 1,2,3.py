# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 14:25:20 2018

@author: Hemant Rathore
"""



## Data Science with Python - Machine Learning Lab


## 1. Regression Case Study -

## A research was conducted recently by state government to collect some health specific statistics 
## for different cities from 5 different states, the objective of the research was to collect the health and 
## medical facilities specific attributes of the cities which directly/indirectly impact the overall health 
## status of the city.

## the objective is to find the attributes and their relation with death rate in order to take 
## the necessary steps to prevent higher death rate.


#### 1. Data Processing ####
#### 1.1 Get the data
### data : health_data for different cities of 5 states

import os
import sys

os.path.dirname(sys.executable)

os.getcwd()

os.chdir("D:\Data Science Training\Data")

import pandas as pd

import numpy as np

data = pd.read_csv("Health_data_new.csv")



## Case Study 1 - Predict death_rate
### dependent variable : death_rate
### independent variables : doctor_availability_rate,hospital_availability_rate,
### annual_per_capita,population_density, State



data.info()

data.describe()


### 1.2 Null Data Handling



pd.isna(data)

pd.isna(data.death_rate)

pd.isna(data.death_rate).value_counts()



## Handling missing values 

data.mean()

data.mean()['death_rate']

data['death_rate'] = data['death_rate'].fillna(data.mean()['death_rate'])


pd.isna(data.death_rate).value_counts()





#### 1.2  Categorical Data Handling using One hot Encoding


object_cols = list(data.select_dtypes(include=['category','object']))

object_cols

data.State.value_counts()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le = LabelEncoder()

ohe = OneHotEncoder(sparse=False)


State_le = le.fit_transform(data.State)

State_le

State_le = State_le.reshape(len(State_le),1)


State_ohe = ohe.fit(State_le)

State_New = State_ohe.transform(State_le)

State_New



State_New = pd.DataFrame(State_New)

State_New.columns = ['s1','s2','s3','s4','s5']


data = data.join(State_New[['s1','s2','s3','s4']])

data.info()

data.drop('State',axis=1,inplace=True)





#### 1.3 Splitting data into Training and Test Data Sets

import seaborn as sns

sns.distplot(data.death_rate)


import sklearn

from sklearn.model_selection import train_test_split


data.info()

data_X = data.iloc[:,[0,1,2,3,4,6,7,8,9]]

data_y = data[:]['death_rate']


X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=100)

X_train.info()

y_train

X_test.info()

y_test

sns.distplot(data.death_rate)

sns.distplot(y_train)

sns.distplot(y_test)


#### 2 building ML Models

## 2.1 Multiple Linear Regression

### dependent variable : death_rate
### independent variables : doctor_availability_rate,hospital_availability_rate,
### annual_per_capita,population_density, state (s1,s2,s3,s4,s5)

## Death Rate(y) = B0 + B1*doctor_availability_rate + B2*hospital_availability_rate +
## B3*annual_per_capita + B4*population_density + B5*s1 + B6*s2 +B7*s3 + B8*s4


## Feature Selection - Checking Correlation to get perfect set of independent variables



import seaborn as sns

sns.pairplot(data)

cr = data.corr()

cr['death_rate']

sns.heatmap(cr,annot=True,cmap="coolwarm")


## Conclusion - Selected independent variables :doctor_availability_rate, hospital_availability_rate,
### annual_per_capita,population_density



X_train.info()

X_train.drop(['City_ID','s1','s2','s3','s4'],axis=1,inplace=True)

X_train.info()

X_test.info()

X_test.drop(['City_ID','s1','s2','s3','s4'],axis=1,inplace=True)

X_test.info()


## Check for Linear Relation

import seaborn as sns; sns.set(color_codes=True)

ax = sns.regplot(x=X_train.doctor_availability_rate, y=y_train)

ax = sns.regplot(x=X_train.annual_per_capita, y=y_train)

ax = sns.regplot(x=X_train.population_density, y=y_train)

ax = sns.regplot(x=X_train.hospital_availability_rate, y=y_train)


## building linear Model

from sklearn import linear_model

lr = linear_model.LinearRegression()

# Train the model using the training sets

lr.fit(X_train,y_train)

lr.intercept_

lr.coef_

lr._residues


## Predicting death_rate for test dataset using model


y_pred = lr.predict(X_test)

res = pd.DataFrame({'y_act':y_test,'y_pred':y_pred})



## Evaluate your model using RMSE

import math

math.sqrt(((y_test-y_pred)**2).mean())



## Analyze your model performance visually


sns.kdeplot(data=res.y_act,data2=res.y_act,shade=True,cmap="gnuplot")

sns.kdeplot(data=res.y_act,data2=res.y_pred,shade=True,cmap="gnuplot")



import matplotlib.pyplot as plt

from matplotlib.pyplot import figure


res = res.sort_values(by='y_act')

x1 = np.arange(1,len(y_test)+1)
y1 = res.y_act


y2 = res.y_pred

figure( figsize=(10, 8))
plt.plot(x1, y1)

figure( figsize=(10, 8))
plt.plot(x1,y2)

figure( figsize=(15, 12))
plt.plot(x1, y1,x1,y2)

## Analyze the error patterns

figure( figsize=(15, 12))
plt.plot(x1,y1-y2)










## 2. Multiple Logistic Regression

## Some of the cities have been given the status of Special Assistance Area on medical background, the 
## objective is to come up with a model which can be used to make the decision if any new city should
## be given this status based on given health and Medical conditions.


### dependent variable : special_assistance_area
### independent variables : doctor_availability_rate,hospital_availability_rate,
### annual_per_capita,population_density, state,death_rate


## special_assistance_area - ln(p/1-p) = B0 + B1*doctor_availability_rate +
## B2*hospital_availability_rate + B3*annual_per_capita + B4*population_density +
## B5*death_rate + B6*s1 + B7*s2 +B8*s3 + B9*s4


import os
import sys


os.chdir("D:\Data Science Training\Data")

import pandas as pd

import numpy as np

data = pd.read_csv("Health_data_new_saa.csv")


data.info()

data.describe()

data.special_assistance_area.value_counts()


## Missing data Handling

pd.isna(data)

pd.isna(data.death_rate)

pd.isna(data.death_rate).value_counts()


## dropping records with null values


data.dropna(inplace=True)


#### 1.2  Categorical Data Handling 

data.info()

data.State

object_cols = list(data.select_dtypes(include=['category','object']))

object_cols


## special_assistance_area - Using Label Encoder



from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

special_assistance_area_le = le.fit_transform(data.special_assistance_area)


data['label'] = special_assistance_area_le

data.drop('special_assistance_area',axis=1,inplace=True)



## State - Using one hot Encoder


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le = LabelEncoder()

ohe = OneHotEncoder(sparse=False)

State_le = le.fit_transform(data.State)

State_le

State_le = State_le.reshape(len(State_le),1)

State_ohe = ohe.fit(State_le)

State_New = State_ohe.transform(State_le)

State_New

State_New = pd.DataFrame(State_New)

State_New.columns = ['s1','s2','s3','s4','s5']


data.reset_index(inplace=True)

data = data.join(State_New[['s1','s2','s3','s4']])

data.info()

data.drop('State',axis=1,inplace=True)



#### 1.3 Splitting data into Training and Test Data Sets

data.label.value_counts()


from sklearn.model_selection import train_test_split


data.info()

data_X = data.iloc[:,[0,1,2,3,4,5,7,8,9,10]]
data_y = data[:]['label']


X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=100)

X_train.info()

y_train

X_test.info()

y_test

import seaborn as sns

sns.distplot(data.label,kde=False)

sns.distplot(y_train,kde=False)

sns.distplot(y_test,kde=False)



## Feature Selection - Checking Correlation to get perfect set of indipendent variables


cr = data.corr()

cr['label']

sns.heatmap(cr,annot=True,cmap="coolwarm")


## Conclusion - Selected independent variables : doctor_availability_rate,hospital_availability_rate,
### annual_per_capita,population_density,death_rate
## building linear Model


X_train.info()

X_train.drop(['City_ID','s1','s2','s3','s4'],axis=1,inplace=True)

X_train.info()

X_test.info()

X_test.drop(['City_ID','s1','s2','s3','s4'],axis=1,inplace=True)

X_test.info()


## Check for logistic Relation

import seaborn as sns; sns.set(color_codes=True)

ax = sns.regplot(x=X_train.doctor_availability_rate, y=y_train,order=3)

ax = sns.regplot(x=X_train.annual_per_capita, y=y_train,order=3)

ax = sns.regplot(x=X_train.population_density, y=y_train,order=3)

ax = sns.regplot(x=X_train.hospital_availability_rate, y=y_train,order=3)

ax = sns.regplot(x=X_train.death_rate, y=y_train,order=3)


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

# Train the model using the training sets

lr.fit(X_train,y_train)

lr.intercept_
lr.coef_



## Predicting death_rate for test dataset using model

y_pred = lr.predict(X_test)

prob = lr.predict_proba(X_test)

res = pd.DataFrame({'y_act':y_test,'y_pred':y_pred,'Prob1':prob[:,1]})

res = res.sort_values(by='Prob1')


## Analyze your model performance visually

## Model Evaluation - Ploting ROC



from sklearn.metrics import roc_curve


fpr, tpr,thresholds = roc_curve(y_test, prob[:,1])


import matplotlib.pyplot as plt

from matplotlib.pyplot import figure


figure( figsize=(10, 8))
plt.plot(fpr, tpr, label='ROC')
plt.plot([0, 1], [0, 1], 'k--')



from sklearn import metrics

AUC = metrics.auc(fpr, tpr)

AUC


## Model Evaluation - Confusion Matrix

from sklearn.metrics import confusion_matrix


cnf_matrix = confusion_matrix(y_test, y_pred)

print(cnf_matrix)

acc = (cnf_matrix[0,0]+cnf_matrix[1,1])/(cnf_matrix[0,0]+cnf_matrix[0,1]+cnf_matrix[1,0]+cnf_matrix[1,1])

 






## 3. KNN Classification

## Case Study -
### Data : Prostate Cancer data

## One of the leading Cancer Research institute is working on a research project to 
## understand the structural properties of various Prostate Cancer tumors and make some 
## conclusion to distinguish between Malignant and Benign Tumors based on the structural differences
## The main objective is to use these properties to classify the tumors into correct categories 
## (Malignant and Benign).



### Dependent variable - Diagnosis_result (Type : Malignant (M) or Benign (B) cancer)
### Independent variables - All other




os.chdir("D:\Data Science Training\Data")

import pandas as pd

import numpy as np

data = pd.read_csv("Prostate_Cancer.csv")


data.info()

data.describe()

data.diagnosis_result.value_counts()

## Missing data Handling

pd.isna(data)

pd.isna(data.diagnosis_result)

pd.isna(data.diagnosis_result).value_counts()




#### 1.2  Categorical Data Handling 

data.info()


object_cols = list(data.select_dtypes(include=['category','object']))

object_cols

data.diagnosis_result.value_counts()


## Using Label Encoder



from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()


label = le.fit_transform(data.diagnosis_result)

label

data = data.join(pd.DataFrame({'label':label}))

data.drop('diagnosis_result',axis=1,inplace=True)


data.info()



#### 1.3 Splitting data into Training and Test Data Sets


data.label.value_counts()


from sklearn.model_selection import train_test_split


data.info()

data_X = data.iloc[:,[0,1,2,3,4,5,6,7,8]]
data_y = data[:]['label']


X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3, random_state=100)

X_train.info()

y_train

X_test.info()

y_test


sns.distplot(data.label,kde=False,bins=3)

sns.distplot(y_train,kde=False,bins=3)

sns.distplot(y_test,kde=False,bins=3)




## Feature Selection - Checking Correlation to get perfect set of indipendent variables


cr = data.corr()

cr['label']

sns.heatmap(cr,annot=True,cmap="coolwarm")


## Conclusion - Selected independent variables : radius,perimeter,area,smoothness,compactness,symmetry
## building linear Model


X_train.info()

X_train.drop(['id','texture','fractal_dimension'],axis=1,inplace=True)

X_train.info()

X_test.info()

X_test.drop(['id','texture','fractal_dimension'],axis=1,inplace=True)

X_test.info()


## Check for Classification boundries


import matplotlib.cm as cm
colors = cm.rainbow(y_train*500)


fig,axes = plt.subplots(figsize=(10,10))
axes.scatter(X_train.symmetry,X_train.area,c=colors,alpha=0.7,s=1000)



fig,axes = plt.subplots(figsize=(10,10))
axes.scatter(X_train.perimeter,X_train.area,c=colors,alpha=0.7,s=1000)


## Build the Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)


# Train the model using the training sets

knn.fit(X_train,y_train)



## Predicting death_rate for test dataset using model

y_pred = knn.predict(X_test)

res = pd.DataFrame({'y_act':y_test,'y_pred':y_pred})




## Analyze your model performance visually


import matplotlib.cm as cm
colors1 = cm.rainbow(y_test*500)
colors2= cm.rainbow(y_pred*500)


fig,axes = plt.subplots(figsize=(10,10))
axes.scatter(X_test.symmetry,X_test.area,c=colors1,s=600,edgecolor='black')
axes.scatter(X_test.symmetry,X_test.area,c=colors2,s=300)



fig,axes = plt.subplots(figsize=(10,10))
axes.scatter(X_test.perimeter,X_test.area,c=colors1,s=500,edgecolor='black')
axes.scatter(X_test.perimeter,X_test.area,c=colors2,s=300)



## Model Evaluation - Confusion Matrix

from sklearn.metrics import confusion_matrix


cnf_matrix = confusion_matrix(y_test, y_pred)



print(cnf_matrix)


acc = (cnf_matrix[0,0]+cnf_matrix[1,1])/(cnf_matrix[0,0]+cnf_matrix[0,1]+cnf_matrix[1,0]+cnf_matrix[1,1])