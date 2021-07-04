# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 10:58:15 2018

@author: Hemant Rathore
"""


### Pandas 


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


import numpy as np

import pandas as pd



df1=pd.DataFrame(data=[[1,2,3,4,5],[6,7,8,9,0]],columns = ["a","b","c","d","e"])

df1

# Using dectionary  


df2=pd.DataFrame({'id':[1,2,3,4],'name':['alex','bella','john','joe'],'year':[2001,2002,2003,2004]})

df2



# Accessing DF elements


# Accessing columns

print(df2)

df2['id']

df2['name']

#select table.col from table

df2.id

df2.name

df2[['id','name']]


# Accessing Records

df2[0:1]

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


df2.iloc[1:3,:]



# Sorting on columns


df2


df2.sort_values('name')


df2.sort_values('name',ascending=False,inplace= True)

df2


## Adding a new column

country =  ['IND','US','EUR','AUS']

country

df2['country'] = country

df2['test'] = 1

df2


## Dropping a column


df2.drop('test',axis=1)

df2.drop('test',axis=1,inplace=True)

df2





## renaming columns

df2.columns = [ 'ID', 'Name', 'Year','Country']

df2

df2.columns = [ 'id', 'name', 'year','country']

# to rename selected columns

df2.rename(columns={'id':'ID','name':'Name'},inplace =True)


pd.columns(df2)

df2.columns = [ 'id', 'name', 'year','country']


## adding a row

df0 = pd.DataFrame({'country':['abc'],'id':[5],'name':['new'],'year':[2018]})

df2 = df2.append(df0)

df2 = df2.append(df0,ignore_index=True)



## Dropping a row

df2

df2.drop(1,axis=0,inplace=True)



## Finding duplicate records

df2.duplicated()


df2

df2.drop_duplicates(inplace=True)

df2


## Conditional data filtering

# select col from table where col=123

# table[(where)],[[select]]

    
# df2[where][select]


df2[df2.id > 2]

df2[df2.id >2 ]['name']

df2[df2.id >2 ][['name','country']]
 
df2[(df2.id != 2) & (df2.year >= 2003)][['name','country']]

df2[(df2.id >2) | (df2.year >= 2001)][['name','country']]

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



## Aggregation operations on DataFrames

df_5 = pd.DataFrame({'id':[1,2,3,4,5],'names':['A','B','C','D','E'],'marks':[87,89,89,72,92],'subject':['S1','S2','S2','S1','S1']})

df_5

df_5.marks.mean()

df_5.marks.std()

df_5.marks.sum()

df_5.marks.min()

df_5.marks.max()

df_5.marks.var()

df_5.marks.median()

df_5.marks.mode()

# Group by

grp = df_5.groupby('subject')

grp.mean()

grp.mean()['marks']

grp.min()['marks']



grp = df_5.groupby(by=['subject','names'])

grp.mean()

# Aggregated summary

grp = df_5.groupby('subject')

grp.describe()['marks']


## Importing data from file



Data = pd.read_csv("D:/Data Science Training/Data/Credit-Scoring-Clean.csv")

Data = pd.read_csv("D:/Data Science Training/Data/Credit-Scoring-Clean.csv",sep=',')


Data

# Analyzing the dataframe

Data.info()

Data.describe()

Data.shape

Data.columns

Data.count()

Data.head(10)

Data.tail(10)

Data.sample(10)

Data.sample(frac=.01)


# writing to file


Data.to_csv("D:/Data Science Training/Data/new_file.csv",index=False)


data_1 = pd.read_excel("D:/Data Science Training/Data/Credit-Scoring-Clean.xlsx",sheet_name='Credit-Scoring-Clean')

data_1

data_1.to_excel("D:/Data Science Training/Data/new_file.xls")




## Connecting to Oracle database and reading data from table



from sqlalchemy import create_engine

conn = create_engine('oracle+cx_oracle://hemant:password@127.0.0.1:1521/?service_name=xe')


pd.read_sql_table('sales', conn)

res = pd.read_sql_query('select * from sales where product_type=\'Eyewear\' and product=\'Bella\' and order_method_type=\'Telephone\'',conn)


# exporting Dataframe to table

res.to_sql('sales_result', conn, if_exists='replace')

