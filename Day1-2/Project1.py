#Data Visualization
import pandas as pd
import os 
os.chdir('D:\\Python A-Z\\Module 1 - Python Basics\\')

sample_super_store=pd.read_excel('Sample - Superstore.xls')

#jointplots

from matplotlib import pyplot as plt
import seaborn as sns

jp1 = sns.jointplot(data=sample_super_store, x = 'Sales', y = 'Profit')

#Histograms

m1 = sns.distplot(sample_super_store.Sales, bins=20)

m1 = sns.distplot(sample_super_store.Profit, bins=20)

m1=sns.regplot(sample_super_store.Sales, sample_super_store.Profit, data=sample_super_store)

n1 = plt.hist(sample_super_store.Profit, bins=15)
n1 = plt.hist(sample_super_store.Sales, bins=15)




sample_super_store[sample_super_store.Segment=='Consumer'].Sales

plt.hist(sample_super_store[sample_super_store.Segment=='Consumer'].Profit, bins=15)
plt.hist(sample_super_store[sample_super_store.Segment=='Corporate'].Profit, bins=15)
plt.hist(sample_super_store[sample_super_store.Segment=='Home Office'].Profit, bins=15)
plt.show()

plt.hist([sample_super_store[sample_super_store.Segment=='Consumer'].Profit, 
          sample_super_store[sample_super_store.Segment=='Corporate'].Profit,
          sample_super_store[sample_super_store.Segment=='Home Office'].Profit],bins=15, stacked=True)


sns.set(color_codes=True)
m1 = sns.distplot(sample_super_store.Profit, bins=20, color='r')

ax=sns.boxplot(x='Segment', y="Sales", data=sample_super_store)
