# -*- coding: utf-8 -*-
"""

@author: Hemant Rathore
"""


### NumPy Arrays - Multidimensional array specially designed for scientific comutations.

 
# Creating NP Array using list

import numpy as np  


my_list = [1,2,3]

np_arr=np.array(my_list)


print(np_arr)

type(np_arr)

print(np_arr.dtype.name)


my_list = ['a','b','c']

np_arr=np.array(my_list)

print(np_arr)

print(np_arr.dtype.name)

my_list = [1,2,3]

np_arr=np.array(my_list, dtype= str)

print(np_arr)

np_arr=np.array(my_list, dtype= float)

print(np_arr)

print(np_arr.dtype.name)


#two dimensional array using List

my_new_list= [[1,2,3],[4,5,6],[7,8,9]]

type(my_new_list)

np_mat = np.array(my_new_list)

print(np_mat)

type(np_mat)

#two dimensional array using tuple

my_tup= ([1,2,3],[4,5,6],[7,8,9])

print(my_tup)
type(my_tup)


np_mat = np.array(my_tup)

print(np_mat)

type(np_mat)


## some inbuilt numpy utilities

# numpy.arange() - to Generate array of sequences

np_seq = np.arange(0,10)

print(np_seq)
   
 # using custom sequences
   
np.arange(0,10,2)
   
np.arange(0,50,3)


# linspaces() - tp devide the range into given number of equal intervals

np.linspace(1,10,100)

# full() - Creating np array using given constant

np.full((5,5),0)


np.full((2,5,5),1)
   
   
# Random number generation from uniform distribution (btw 0-1)
   
np.random.rand(1)
   
np.random.rand(100)
  
np.random.rand(5,5)


# Analyzing the Array Structure

# size of array
  
a=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
   
print(np.size(a))
   
print(a.size)
  
print(len(a))


# Subsetting & Slicing operations
   
a=np.array([(1,2,3),(4,5,6),(7,8,9),(10,11,12)])
 
a
 
a[0,0]
 
a[0,2]
 
a[0,:]

a[0]
 
a[:,0] 

a[0:2,2]
 
a[0:2,1:3]



## Array Manipulation

a=np.array([1,2,3,4,5,6,7,8,9,10,11,12])

# Structure of array 

print(a.shape)

print(a.nbytes)
  
# Reshape array

new_array = a.reshape(3,4)

print(new_array)

print(new_array.shape)

new_array = a.reshape(3,4,order='F')

print(new_array)

print(new_array.shape)


# Transpose Array

a=np.array([(1,2,3),(4,5,6),(7,8,9),(10,11,12)])
 
print(a)

print(np.transpose(a))

# Insert new element

a=np.array([1,2,3,4,5,6,7,8,9,10,11,12])

np.insert(a,12,13)

# delete the element

a
np.delete(a,11)


# Append array


b=np.array([13,14,15])

np.append(a,b)


# Copy Array

a

b=a

b

a[0]=100

a

b

c= a.copy()

c

a[0]=1

a

c

b


## Mathematical functions of Array

a=np.array([1,3,2,4,7,5,8,11,12,9,10,6])
   

# max
a.max()
# min
a.min()
# sum
a.sum()
# mean
a.mean()
# median
np.median(a)
# Std. Dev.
a.std()
# variance
a.var()

# sort
a
a.sort()
a


# using 2d array and their operations
  
a=np.array([(1,2,3),(4,5,6)])
 
a
 
# col sum

print(a.sum(axis=0))
 
# row sum
 
print(a.sum(axis=1))
 
 


# Arithmetic Operations on Array
 
a=np.array([(1,2,3),(4,5,6)])

b=np.array([(1,2,3),(4,5,6)])
 
a

b

# addition

print(a+b)

# substraction

print(a-b)

# multiplication

print(a*b)

# division

print(a/b)
   

# Matrix Multiplication
   

b=np.array([(1,2,3),(4,5,6),(7,8,9)])

a
b

a*b
  
np.matmul(a,b)


  
## Vertical and Horizontal stack (row and column wise binding)

a=np.array([1,2,3,4,5,6,7,8,9,10,11,12])

b=np.array([10,20,30,40,50,60,70,80,90,100,110,120])
 

a

b

print(np.vstack((a,b)))

print(np.row_stack((a,b)))

# horizontal

print(np.hstack((a,b)))
print(np.append(a,b))

# Column Wise concatenate

print(np.column_stack((a,b)))

# Vertical and Horizontal split

a=np.array([(1,2,3,4),(5,6,7,8)])

a

a1,a2 = np.vsplit(a,2)

a1

a2

a1,a2 = np.hsplit(a,2)

a1

a2

# Read and Wrtie from/to file



np.savetxt("array.csv", a, delimiter=";") # how to store in int format

b= np.loadtxt("array.csv",delimiter=";", dtype=int)

b







### 6 Pandas 


## Series - One dimensional array

import numpy as np

import pandas as pd



sr = pd.Series(data= [1,2,3,4])

sr
    
# Labeling the elements

label = ['a','b','c','d']

values = [1,2,3,4] 


sr = pd.Series(data=values,index=label) 

sr

type(sr)

# NP Array to Series

a= np.array(values)

a

sr = pd.Series(a)

sr

# Dict. to Series
  
my_dict= {'a':1,'b':2,'c':3,'d':4}

sr = pd.Series(my_dict)
 
sr

# Arithametic Operations on Series

sr+sr

sr*2

sr/sr

sr-sr






## Data frame - Real world 2D RDBMS type tables 

# function - DataFrame()

# directly using the data or list


df1=pd.DataFrame(data=[[1,2,3,4,5],[6,7,8,9,0]],columns = ["a","b","c","d","e"])

df1

# Using dectionary  

df2=pd.DataFrame({'id':[1,2,3,4],'name':['alex','bella','john','joe'],'year':[2001,2002,2003,2004]})

df2


my_dict= {'id':[1,2,3,4],'car':['maruti','honda','jeep','dtsun']} 
  
df3= pd.DataFrame(my_dict)

df3


# Accessing DF elements

df2

# Accessing columns

df2['id']

df2['name']

df2.id

df2.name

df2[['id','name']]

# Accessing Records

df2[0:2]

df2[0:]

df2[:2]


## or

df2.iloc[0]

df2.iloc[1:3]


# Accessing Columns and records together

df2[0:2]['name']

df2[0:2][['id','name']]

# or

df2.iloc[[0],[0]]

df2.iloc[[0,1],[0,1]]


df2.iloc[:,[0,1]]


df2.iloc[[0,1],:]


df2.iloc[1:3,1:3]

## Adding a new column

country =  ['IND','US','EUR','AUS']

country

df2['country'] = country

df2['test'] = 1

df2


## Dropping a column


df2.drop('test',axis=1)

df2.drop('test',axis=1)

df2.drop('test',axis=1,inplace=True)

df2


## adding a row

df0 = pd.DataFrame({'country':['abc'],'id':[5],'name':['new'],'year':[2018]})

df2 = df2.append(df0,ignore_index=True)


# reseting the indexes

df2.reset_index(drop=True,inplace=True)


help(df2.append)

## Dropping a row

df2

df2.drop(1,axis=0,inplace=True)

df2

## Conditional data filtering


# table[(where)],[[select]]


df2[df2['id']>2]

df2[df2['id']>2]['name']

df2[df2['id']>2][['name','country']]
 
df2[(df2['id']>2) & (df2['year']>=2003)][['name','country']]

df2[(df2['id']>2) | (df2['year']>=2001)][['name','country']]

df2


## Merging/joining dataframes

# Inner join

df_1=pd.DataFrame({'id':[1,2,3],'name':['alex','bella','john'],'year':[2001,2002,2003]})
df_1
df_2=pd.DataFrame({'id':[1,2,4], 'city':['A',' B','C'],'pin':[1002002,1002003,1002001]})
df_2
  
df_merge = pd.merge(df_1,df_2 ,on='id')

df_merge

# outer joins

pd.merge(df_1,df_2,how='outer',on='id')

pd.merge(df_1,df_2,how='left',on='id')

pd.merge(df_1,df_2,how='right',on='id')


# DataFrames Binding

# Row wise

df_1

df_3=pd.DataFrame({'id':[4,5,6],'name':['alex1','bella1','john1'],'year':[2001,2002,2003]})

df_3


pd.concat([df_1,df_3],axis=0)


# Column wise - Not preferred

df4 = pd.concat([df_1,df_2],axis=1)

df4 = pd.concat([df_1,df_2],axis=1,verify_integrity=True)


## Aggregation operations on DataFrames

df_5 = pd.DataFrame({'id':[1,2,3,4,5],'names':['A','B','C','D','E'],'marks':[87,89,65,72,92],'subject':['S1','S2','S2','S1','S1']})

df_5

df_5.marks.mean()

df_5.marks.std()

df_5.marks.sum()

df_5.marks.min()

df_5.marks.max()

df_5.marks.var()

df_5.marks.median()

# Group by

grp = df_5.groupby('subject')

grp.mean()

grp.mean()['marks']

grp.max()['marks']

# aggregated summary

grp.describe()['marks']


## Importing data from file

Data = pd.read_csv("D:/Data Science Training/Data/Credit-Scoring-Clean.csv")

Data

# Analyzing the dataframe

Data.info()

Data.describe()

Data.shape

Data.columns

Data.count()



# writing to file

Data.to_csv("D:/Data Science Training/Data/new_file.csv",index=False)


data_1 = pd.read_excel("D:/Data Science Training/Data/Credit-Scoring-Clean.xlsx",sheet_name='Credit-Scoring-Clean')

data_1

data_1.to_excel("D:/Data Science Training/Data/new_file.xls")



