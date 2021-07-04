
#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Created on Mon Jul 16 12:08:41 2018

@author: hr250012

"""
## Advanced & Interactive Visualization with Plotly


## Ploting Steps
# 1. Create plot trace
# 2. Creating data and layout objects
# 3. Creating plot object
# 4. draw the plot


import plotly

import plotly.plotly as py

from plotly.offline import plot

plotly.__version__


import plotly.graph_objs as go

import numpy as np

import pandas as pd

 

## Line & Point Plot

# Create X, Y Coord.

random_x = np.linspace(start = 0, stop = 20, num = 100)

random_y = np.linspace(start = 0, stop = 20, num = 100)+ np.random.randn(100)


# 1. Create a trace - it is used to define graph and plotting properties

trace = go.Scatter(

    x = random_x,

    y = random_y,

    mode = 'markers' # markers mode for point plot, lines for line

)

 

#2.  set data and layout

 
data = [trace]

layout = go.Layout(title="Line & Point Plot")


#3. Create Plot object

plot = dict(data = data,layout=layout)
 

#4. Draw the plot

plotly.offline.plot(plot,filename='Line-Point-Plot')

 



# Bar plots


# Using DF

df2= pd.DataFrame({'Company':['A & B','Blue Diamond','CC Trading','Dog Club','E-V'],'Revenue':[100,56,150,12,134]})


trace = go.Bar(

    x = df2['Company'],

    y = df2['Revenue'],
    
    marker=dict(
    color = 'green'),
    opacity=0.6

)

data = [trace]


layout = go.Layout(title="Revenue Report - Q2-2018",width=700,height=700)


plot = dict(data = data,layout=layout)


plotly.offline.plot(plot,filename='Revenue Report')

 
# Histogram plots

 
Sales = np.random.randn(100)*15+100


df = pd.DataFrame({'Sales' : Sales})


trace = go.Histogram(

    x = df.Sales,

    marker=dict(

        color = 'rgb(44, 160, 101)',

        line = dict(width = 1, color='gray')


    )


)


data = [trace]

layout = go.Layout(title="Sales Distribution",width=600,height=600,
        xaxis= dict(
        title= 'Sales'
    ),
    yaxis=dict(
        title= 'Product Count'
    ))


plot = dict(data = data,layout=layout)


plotly.offline.plot(plot,filename='HistoGram')

 

 
# Box plot

import random

np.random.seed(1021)

Revenue = np.random.randn(100)*100+1000

Industry = (np.repeat(random.sample(['IT','BPO','Electronic','Communication'],4),25))

df1= pd.DataFrame({'Revenue' : Revenue,'Industry' : Industry})

 

trace = go.Box(

    y = df1.Revenue,

    x = df1.Industry,

    marker=dict(size =15, # outlier size

        outliercolor='red',
        color='gray'

                )
                )


data = [trace]


layout = go.Layout(title="Revenue distribution by Industries",width=800,height=800,
        
        xaxis= dict(
        title= 'Industry'
    ),
    yaxis=dict(
        title= 'Revenue'
    ))


plot = dict(data = data,layout=layout)


plotly.offline.plot(plot,filename='BoxPlot')

 


## Scatter Plot


# Create X, Y Coord.

 
Company_Size = np.random.randn(200)*20+200

Company_Age = np.random.randn(200)*2+15

 
# Create a trace

 
trace = go.Scatter(

    x = Company_Size,

    y = Company_Age,

    mode = 'markers' # markers mode for point plot

)

 
# set data and layout

 
data = [trace]


layout = go.Layout(title="Company by firm size and age",
        xaxis= dict(
        title= 'Company Size'
    ),
    yaxis=dict(
        title= 'Company Age'
    ))


plot = dict(data = data,layout=layout)


plotly.offline.plot(plot,filename='Scatter-Plot')



 
## Some more Customization using marker and layout

import random
 
np.random.seed(1001)

Company_Size = np.random.randn(400)*20+200

np.random.seed(1021)

Company_Age = np.random.randn(400)*3+15

Tier = np.repeat(random.sample([1,2,3,4,5,6,7,8,9,10,1],10),40)

Revenue = np.repeat(random.sample([1,2,3,4,5,6,7,8,9,10,1],10),40)+10

 
# Create a trace

 
trace = go.Scatter(

    x = Company_Size,

    y = Company_Age,

    mode = 'markers',

    marker=dict(

        size=Revenue,

        color = Tier,

        colorscale  = 'Greens',

        line = dict(width = 1, color='gray'),

        showscale=True

    )

)

# set data and layout


data = [trace]


layout = go.Layout(title="Company Distribuction by firm Size(X), Age(Y), Business Tier (Color) & Revenue (Size)",
        xaxis= dict(
        title= 'Company Size'
    ),
    yaxis=dict(
        title= 'Company Age'
    ),
    plot_bgcolor='black',
    width=800,height=800

                   )

plot = dict(data = data,layout=layout)

plotly.offline.plot(plot,filename='Scatter-Plot')

 
# some more colorscale values - Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis, Cividis



## Multiple Plots Together


# Data

 
Month_100 = np.linspace(start = 1, stop = 100, num = 100)

Fund1 = np.linspace(start = 5, stop = 20, num = 100)+ np.random.randn(100)

Fund2 = np.linspace(start = 10, stop = 20, num = 100)+ np.random.randn(100)

sensex =  np.random.randn(100)



















# Solution


# Create traces

 
trace1 = go.Scatter(

    x = Month_100,

    y = Fund1,

    mode = 'markers+lines',

    name = 'Fund 1'

)



trace2 = go.Scatter(

    x = Month_100,

    y = Fund2,

    mode = 'lines',

    name = 'Fund 2'

)



trace3 = go.Scatter(

    x = Month_100,

    y = sensex,

    mode = 'markers',

    name = 'Sensex % change MoM'

)


# set data and layout


data = [trace1,trace2,trace3]


plotly.offline.plot(data,filename='Multiple-Plots')








# for online plotly use

py.plot(plot, filename = 'Scatter-Plot')

