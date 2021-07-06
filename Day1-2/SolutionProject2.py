#Importing data from excel or csv
import pandas as pd
import os 
os.chdir('D:\\Python A-Z\\')

sample_super_store=pd.read_excel('Sample - Superstore.xls')

"""
SELECT Ship Mode, Segment, Region, City
FROM sample_super_store
LIMIT 5;
"""
sample_super_store[['Ship Mode', 'Segment', 'Region', 'City']].head(5)

"""
SELECT *
FROM sample_super_store
WHERE Segment = 'Consumer'
LIMIT 5;
"""
sample_super_store[sample_super_store['Segment'] == 'Consumer'].head(5)

"""
SELECT *
FROM sample_super_store
WHERE Segment = 'Consumer' AND Sales > 500.00;
"""
sales_500=sample_super_store[(sample_super_store['Segment'] == 'Consumer') & (sample_super_store['Sales'] > 500.00)]

"""
SELECT *
FROM sample_super_store
WHERE Segment = 'Consumer' OR Sales > 500;
"""
sample_super_store[(sample_super_store['Segment'] == 'Consumer') | (sample_super_store['Sales'] > 500)]


"""
SELECT Segment, count(*)
FROM sample_super_store
GROUP BY Segment;
"""
sample_super_store.groupby('Segment').size()

sample_super_store.groupby('Segment').count()

sample_super_store.groupby('Segment')['Sales'].count()

"""
SELECT Segment, AVG(Sales), COUNT(*)
FROM sample_super_store
GROUP BY Segment;
"""

import numpy as np

sample_super_store.groupby('Segment').agg({'Sales': np.mean, 'Segment': np.size})

"""
SELECT Region, Segment, COUNT(*), AVG(Sales)
FROM sample_super_store
GROUP BY Region, Segment;
"""

sample_super_store.groupby(['Region', 'Segment']).agg({'Sales': [np.size, np.mean]})