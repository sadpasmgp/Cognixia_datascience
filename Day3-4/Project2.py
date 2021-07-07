import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Cities = pd.read_csv('path and file name')
Cities.head(10)

#States Population
color = ['darkred', 'darkgreen', 'silver', 'teal', 'pink', 'orange', 'salmon', 'plum', 'peru']

state_population = Cities[['population_total','state_name']].groupby('state_name').sum().sort_values(['population_total'],ascending=False)

plt.rcParams['figure.figsize'] = 8,4

state_population.plot(kind="bar", color=color)
plt.xlabel('State Name', size=20)
plt.ylabel('Total Population', size=20)
plt.show()

#Analyzing literacy rates
color = ['darkred', 'darkgreen', 'silver', 'teal', 'pink', 'orange', 'salmon', 'plum', 'peru']
literacy_by_state = Cities[['effective_literacy_rate_total','state_name']].groupby('state_name').mean().sort_values(['effective_literacy_rate_total'],ascending=False)
literacy_by_state.plot(kind="bar", color=color)
plt.xlabel('State Name', size=20)
plt.ylabel('Average Literacy Rate', size=20)
plt.ylim(75,100)
plt.show()

#Sex ratio by states
color = ['darkred', 'darkgreen', 'silver', 'teal', 'pink', 'orange', 'salmon', 'plum', 'peru']
literacy_by_state = Cities[['sex_ratio','state_name']].groupby('state_name').mean().sort_values(['sex_ratio'],ascending=False)
literacy_by_state.plot(kind="bar", color=color)
plt.xlabel('State Name', size=20)
plt.ylabel('Average Sex Ratio', size=20)
plt.ylim(800,1100)
plt.show()

#Percentage of graduates against total populkation of states
Cities['graduates_percentage'] = 100 * Cities['total_graduates'] / Cities['population_total']

Cities.head(10)


color = ['darkred', 'darkgreen', 'silver', 'teal', 'pink', 'orange', 'salmon', 'plum', 'peru']
graduate_percentage_by_state = Cities[['graduates_percentage','state_name']].groupby('state_name').mean().sort_values(['graduates_percentage'],ascending=False)
graduate_percentage_by_state.plot(kind="bar", color=color)
plt.xlabel('State Name', size=20)
plt.ylabel('Graduate Percentage', size=20)
plt.ylim(5,30)
plt.show()

#States with most cities in the Top 500
number_of_cities_by_state = Cities[['city','state_name']].groupby('state_name').count().sort_values(['city'],ascending=False)
number_of_cities_by_state.plot(kind="bar", color=color)
plt.xlabel('State Name')
plt.ylabel('Number of Cities')
plt.show()



#Top 10 cities with high literacy
Cities[['city','effective_literacy_rate_total']].sort_values(['effective_literacy_rate_total'],ascending=False).head(10)

