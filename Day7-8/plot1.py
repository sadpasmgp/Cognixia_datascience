import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('/Users/Ganesh/Desktop/Cognixia/Data/gas_prices.csv')
for country in ["Australia", "Canada","France", "Germany","Italy","Japan","Mexico"]:
    plt.plot(df['Year'], df[country])
plt.legend()
plt.show()
