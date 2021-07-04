# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:27:55 2018

@author: Hemant Rathore
"""


import matplotlib.pyplot as plt


import numpy as np


# Using basic plot function

# Generate x, y coordinates

x = np.linspace(start = 0, stop = 20, num = 20)

x

y = np.linspace(start = 0, stop = 20, num = 20)+ np.random.randn(20)
 
y

# Draw the Plot

plt.plot(x, y)
 

# Some basic customizations

plt.plot(x,y,'g^--') ## https://matplotlib.org/api/markers_api.html


plt.plot(x, y, color='green', linestyle='solid', marker='^')



plt.plot(x, y, color='green', linestyle='solid', marker='o',
         markerfacecolor='red', markersize=10,linewidth=3,alpha=0.5)

plt.xlabel('X')

plt.ylabel('Y')

plt.title('Title')  





# multiple plots together

 
x1 = np.linspace(start = 0, stop = 20, num = 100)
y1 = np.linspace(start = 0, stop = 20, num = 100)+ np.random.randn(100)*2+5

 
x2 = np.linspace(start = 0, stop = 20, num = 100)
y2 = np.linspace(start = 0, stop = 20, num = 100)+ np.random.randn(100)*2+10

 
plt.plot(x, y, x1, y1,x2,y2)

 
## The Perfect way to use Matplotlib 


# Create fig and axes objects using subplots function

fig,axes = plt.subplots()

# Call plot function using axes objects

axes.plot(x,y,label='Fund 1')
axes.plot(x1,y1,label='Fund 2')
axes.plot(x2,y2,label='Fund 3')

axes.legend()



# It allows you to have multiple subplots too

fig,axes = plt.subplots(nrows=3,ncols=1,figsize=(10,10))

axes[0].plot(x,y)
axes[1].plot(x1,y1)
axes[2].plot(x2,y2)



# Some Cutomization:

fig,axes = plt.subplots(2,2,figsize=(10,10))

axes[0,0].plot(x,y,color='green')
axes[0,0].set_title('Title 1')
axes[0,0].set_xlabel('X1')
axes[0,0].set_ylabel('Y1')
axes[0,1].plot(x1,y1,color='red')
axes[0,1].set_title('Title 2')
axes[0,1].set_xlabel('X2')
axes[0,1].set_ylabel('Y2')
axes[1,0].plot(x,y**2,color='blue')
axes[1,0].set_title('Title 3')
axes[1,0].set_xlabel('X3')
axes[1,0].set_ylabel('Y3')
axes[1,1].plot(x2,y2,color='orange')
axes[1,1].set_title('Title 4')
axes[1,1].set_xlabel('X4')
axes[1,1].set_ylabel('Y4')
plt.tight_layout()

axes[0,0].grid(True)
axes[0,1].grid(True)
axes[1,0].grid(True)
axes[1,1].grid(color='r', alpha=0.5, linestyle='--', linewidth=0.8)


# saving the plot to file

fig.savefig('name.pdf',dpi=200)

# Different Type of plots

# EDA - Univariate Analysis

## Box plot


Salary_India = np.random.randn(10000)*2000 + 25000


fig,axes = plt.subplots(figsize=(10,10))

axes.boxplot([Salary_India],labels=['India'],patch_artist=True)




Salary_US = np.random.randn(10000)*5000 + 83000
Salary_AUS = np.random.randn(10000)*2700 + 66000
Salary_UK = np.random.randn(10000)*3600 + 74000

data = [Salary_India,Salary_US,Salary_AUS,Salary_UK]

labels =['India','US','AUS','UK']

fig,axes = plt.subplots(figsize=(10,10))

axes.boxplot(data,labels=labels,patch_artist=True)



## Histogram


Salary_India = np.random.randn(10000)*2000 + 25000

fig,axes = plt.subplots(figsize=(10,10))

axes.hist(Salary_India,rwidth=0.9,color='g',bins=10)



# EDA - Bi-variate Analysis

## Bar plot


Companies = ['A & B','Blue Diamond','CC Trading','Dog Club','E-V']
Sales = [250,200,140,220,350]
Revanue = [150,110,80,175,250]
 

fig,axes = plt.subplots(1,2,figsize=(10,10))


axes[0].bar(Companies, Sales, label="Sales", color='b')
axes[1].bar(Companies, Revanue, label="Revanue", color='g')

axes[0].legend()
axes[1].legend(loc='upper left')
axes[1].set_xticklabels( Companies, rotation=90 )



## side by Side Bar plot

x_pos = np.array([1,2,3,4,5])

fig,axes = plt.subplots(figsize=(10,10))

axes.bar(x_pos, Sales, label="Sales", width=0.4,color='b')
axes.bar(x_pos+0.3, Revanue, label="Revanue", width=0.4, color='g')

axes.set_xticklabels(Companies,rotation=60)
axes.set_xticks(x_pos+0.2)
axes.legend()


# EDA - Multivariate Analysis

## Sactter Plot/bubble Plot


size = np.random.randn(100)*5+100
years = np.random.randn(100)*1+10


Business = np.random.choice([1,2,3,4,5], size=100, replace=True)
Revenue = (np.random.rand(100)*30)**2

fig,axes = plt.subplots(figsize=(8,8))


axes.scatter(size,years,c=Business,s=Revenue,alpha=0.7)

axes.set_xlabel('size')

axes.set_ylabel('years')
axes.grid()

