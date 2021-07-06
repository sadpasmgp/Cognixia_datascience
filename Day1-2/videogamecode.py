# courtesy ign dataset - Gaming & vik
# 1. Open the dataset and show
'''
In this Python data science tutorial, we’ll use Pandas to analyze video game reviews from IGN, 
a popular video game review site
'''
# Stored in csv format. what is csv format?
The columns contain information about that game:

score_phrase — how IGN described the game in one word. This is linked to the score it received.
title — the name of the game.
url — the URL where you can see the full review.
platform — the platform the game was reviewed on (PC, PS4, etc).
score — the score for the game, from 1.0 to 10.0.
genre — the genre of the game.
editors_choice — N if the game wasn’t an editor’s choice, Y if it was. This is tied to score.
release_year — the year the game was released.
release_month — the month the game was released.
release_day — the day the game was released.

#A DataFrame is a way to represent and work with tabular data — data that’s in table form, 
like a spreadsheet.

 the pandas.read_csv function.
 This function will take in a csv file and return a DataFrame.
 
 Read ign.csv into a DataFrame, and assign the result to a new variable called reviews
 so that we can use reviews to refer to our data.
 
import pandas as pd
reviews = pd.read_csv("ign.csv")

reviews.head()

reviews.shape # num of rows and columns

'''
NOTE: One of the big advantages of using Pandas over a similar Python package like NumPy is that Pandas
 allows us to have columns with different data types.
 
 In our data set, reviews, we have columns that store float values like score, string values like
 score_phrase, and integers like release_year, so using NumPy here would be difficult,
 but Pandas and Python handle it well.
 
 The iloc method allows us to retrieve rows and columns by position. In order to do that, 
 we’ll need to specify the positions of 
 the rows that we want, and the positions of the columns that we want as well.
'''

reviews.iloc[0:5,:] # replicate head()

'''
Let’s dig in a little deeper into our code: we specified that we wanted rows 0:5.
 This means that we wanted the rows from position 0 up to, but not including, position 5.

The first row is considered to be in position 0, so selecting rows 0:5 gives us the rows at
 positions 0, 1, 2, 3, and 4
 
reviews.iloc[:5,:] — the first 5 rows, and all of the columns for those rows.
reviews.iloc[:,:] — the entire DataFrame.
reviews.iloc[5:,5:] — rows from position 5 onwards, and columns from position 5 onwards.
reviews.iloc[:,0] — the first column, and all of the rows for the column.
reviews.iloc[9,:] — the 10th row, and all of the columns for that row.
'''

# INDEXING USING LABELS : DIFF BET LOC AND ILOC

Working with column positions is possible, but it can be hard to keep track of which number 
corresponds to which column.

We can work with labels using the pandas.DataFrame.loc method, which allows us to index using 
labels instead of positions. 

reviews.loc[0:5,:]

# retreive columns by label

reviews.loc[:5,"score"]

reviews.loc[:5,["score", "release_year"]] # more than 1 column

reviews.iloc[:,1] #— will retrieve the second column.
reviews.loc[:,"score_phrase"]# — will also retrieve the second column.

reviews["score"] # same as above

reviews[["score", "release_year"]] # list of columns

type(reviews["score"]) # verifies it is series

reviews["score"].mean()

reviews.mean()  # mean of individual columns

reviews.mean(axis=1) # mean of each row

# axis 0 is for columns

pandas.DataFrame.corr — finds the correlation between columns in a DataFrame.
pandas.DataFrame.count — counts the number of non-null values in each DataFrame column.
pandas.DataFrame.max — finds the highest value in each column.
pandas.DataFrame.min — finds the lowest value in each column.
pandas.DataFrame.median — finds the median of each column.
pandas.DataFrame.std — finds the standard deviation of each column.

reviews.corr()
