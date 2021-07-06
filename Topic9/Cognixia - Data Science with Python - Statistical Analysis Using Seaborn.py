# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 19:45:09 2018

@author: Hemant Rathore
"""

import seaborn as sns

sns.__version__

import pandas as pd

import numpy as np

 
ds = sns.load_dataset('flights')

ds.head()


## Distribution plot, histogram, rug Plot using distplot()

 
sns.distplot(a=ds['passengers'])

 
sns.distplot(a=ds['passengers'],kde=False,bins=25)

  

# Customization using hist_kws,kde_kws


from scipy import stats

sns.distplot(a=ds['passengers'],rug=True,color='b',fit=stats.norm,

             hist_kws={"linewidth": 1,

                            "alpha": 0.5, "color": "g","label": "Hist"},

             kde_kws={"color": "r", "lw": 2, "label": "KDE"} 

                       )

 


## Distribution Plots using kdeplot()


sns.kdeplot(data=ds['passengers'],shade=True)

 
# Cumulative Distribution

 
sns.kdeplot(data=ds['passengers'],shade=True,cumulative=True)

 

# Bivariate Distribution Analysis using jointplot

y = np.random.randn(144)+200

ds=ds.assign(passengers_new=y)




sns.jointplot(x="passengers",y="passengers_new",data=ds,kind="reg")

 
# p value is the probability that you will get this result when there is no correlation present so in case of
# good correlation this should be minimum 

 



## Correlation Analysis using Heatmaps


corr = ds.corr()


sns.heatmap(corr)


sns.heatmap(corr,annot=True)

  
sns.heatmap(corr,annot=True,cmap="coolwarm")


 
#### Statistical Analysis in Python

import numpy as np


# Sampling - with and without replacement


p = [1,2,3,4,5,6,7,8,9,10]

p

np.random.choice(p, size=5)

np.random.seed(10)

np.random.choice(p, size=5)

np.random.seed(10)

np.random.choice(p, size=5)


np.random.choice(p, size=5, replace=False)


## Simulating Standard Normal probability Distribution


from numpy.random import randn

sn_data = randn(1000000)

import seaborn as sns

sns.kdeplot(data=sn_data,shade=True)


## Calculating probability in standard normal distribution

from scipy.stats import norm

norm.pdf(0)

## Cumulative probability in std. normal distribution

sn_data = randn(1000)

sns.kdeplot(data=sn_data,shade=True,cumulative=True)


## Calculating cumulative probability in standard normal distribution


norm.cdf(0)



## Simulating Normal probability Distribution


n_data = randn(100000)*5+100

n_data


sns.kdeplot(data=n_data,shade=True)



## Binimial Distribution Exercise


from scipy.stats import binom


# binom.pmf(x,n,p)

## During the testing for ABC invoice processing system, testing team has observed that
## the chances of invoice failure are 0.012, this system is supposed to process approximately 2500 
## invoices a day, what is the probability of facing 25 failures a day? also calculate cumulative probability 
## of 25 or more failures day?


p = 0.012
n = 2500
x = 25

## P(x) = ncx P^x  (1-P)^n-x  ->  2500 C 25 * 0.012^25  (1-0.012)^2500-25 

## 0.0511 -> ~5%

## Calculating the brobability of getting for 'x' successes out of 'n' trials with 'p' probability


binom.pmf(x,n,p)


## Plotting PMF plot


import matplotlib.pyplot as plt

X = np.arange(binom.ppf(0.001, n, p),binom.ppf(0.999, n, p)) # ppf returns x value for given value of cumulative Probability q

fig, ax = plt.subplots(1, 1,figsize=(10,10))

ax.plot(X, binom.pmf(X, n, p), 'ro')

ax.vlines(x=X, ymin=0,ymax= binom.pmf(X, n, p), colors='b', lw=5, alpha=0.5)


## calculate cumulative probability q to get 25 or more failures


binom.cdf(x,n,p)


1-binom.cdf(x,n,p)



## Poisson Distribution Exercise

## A telecom company has its 24x7 customer care call center, due to limited staff at night the call
## waiting time increases drastically,as per the records the call center recieves approx. 80 calls daily 
## during 12 AM to 6 AM, company is planning to add more staff at night so that they can attend upto 
## 100 calls during this window. What is probability that company may receive even more than 100 calls 
## on any day during the same hrs.

## Probability of 100 or more calls using Poisson Distribution

l = 80

x=100

## P(x) =  (lemda^x  e^-lemda )/x!-> (80^100 * 2.71828^-80 )/100! 

import math

(80**100 * 2.71828**(-80))/math.factorial(100)

## Calculating the brobability of getting 'x' occurances when lambda = l

## poisson.pmf(x,lambda)

from scipy.stats import poisson


poisson.pmf(x,l)


## Plotting PMF plot

import matplotlib.pyplot as plt

X = np.arange(poisson.ppf(0.001, l),poisson.ppf(0.999, l)) # ppf returns x value for given value of cumulative Probability q

fig, ax = plt.subplots(1, 1,figsize=(10,10))

ax.plot(X, poisson.pmf(X, l), 'ro')

ax.vlines(x=X, ymin=0,ymax= poisson.pmf(X, l), colors='b', lw=5, alpha=0.5)


## calculate cumulative probability q to get 100 or more calls

1-poisson.cdf(x,l)


#### Hypothesis testing using 1 sample z Test in R

## scenario : The average marks scored by studenets at XYZ university in year >=2010<= 2016 is 780
## with SD = 16, a sample of 30 student's marks was taken and found average score of these 30
## students is 775, using this sample can we reject the given assumption and conclude that overall average marks is no
## longer = 780?


## Data Points -- Sample mean = 775, n=30,  population mean = 780, population SD = 16



## 1 Define your Null and Alternative hypothesis, H0 & H1

# H0 :  µ = 780
# H1 :  µ != 780


## 2 Define level of significance alpha , generally assumed to be 0.05 or 0.01

alpha = 0.05


## 3 Find the Test Statistics using TS = (X- µ)/ (sigma/ sqrt(n)) or TS = (X- µ)/ (s/ sqrt(n))  {when Ï is not known but n>=30}

# TS = (X- µ)/ (sigma/ sqrt(n))

TS =  (775-780)/(16/math.sqrt(30))

TS #-1.711

from numpy.random import randn

sn_data = randn(1000000)

import seaborn as sns

sns.kdeplot(data=sn_data,shade=True)


## 4 Find P Value using Z Table and TS, if its a two sided test then double P value


from scipy.stats import norm

PV = norm.cdf(TS)*2

PV


## 5 Reject the null hypothesis or you may accept the alternative hypothesis if P < alphs

# p value >alpha (0.05) ---> cannot reject null hypothesis H0



####################################################################

#### Hypothesis testing using 1 sample t Test in python

## scenario : AS per the standards the mean propotion of some kind of bactaria XYZ in a ABC solution 
## is at least  0.95 units, a sample of 20 observations of this solution was taken and found the average 
## proportion of bactaria is 0.85 and SD = 0.16, using this sample can we conclude that average proportion
## is less than the current standard.


## Data Points -- Sample mean = 0.85, sample SD = 0.16, n=20, population mean = 0.95, population SD = NA

## 1 Define your Null and Alternative hypothesis, H0 & H1

# H0 :  µ >= 0.95
# H1 :  µ < 0.95

## 2 Define level of significance alpha , generally assumed to be 0.05 or 0.01

alpha = 0.05


## 3 Calculate Degree of Freedom DF = n-1

DF = 19

DF

## 4 Find the Test Statistics using TS = (X- µ)/ (s/ sqrt(n))

# TS = (X- µ)/ (s/ sqrt(n))

TS =  (0.85-0.95)/(0.16/math.sqrt(20))

TS # -2.795085

from scipy.stats import t

X = np.linspace(t.ppf(0.001, DF),t.ppf(0.999, DF), 100)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

ax.plot(X, t.pdf(X, DF),'r', lw=5)


## 5 Find P Value using t Table, TS and DF, if itâs a two sided test then double P value


PV = t.cdf(TS,DF)

PV #  0.005773304

## 6 Reject the null hypothesis or you may accept the alternative hypothesis if P < alpha

# 0.005773304 << alpha (0.05) ---> can reject null hypothesis H0




## Practice Dataset


data = sns.load_dataset('iris')





