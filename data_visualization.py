import pandas as pd
import matplotlib.pylab as plt
from pandas.plotting import scatter_matrix

#read dataset
df = pd.read_csv('breast-cancer-wisconsin.csv')

#Preprocess dataset
df.replace('?',-99999, inplace=True)
print(df.axes)

df.drop(['id'], 1, inplace=True)

print(df.loc[10])

# Print the shape of the dataset
print(df.shape)

# Describe the dataset
print(df.describe())

# Plot histograms for each variable
df.hist(figsize = (10, 10))
plt.show()

# Create scatter plot matrix
scatter_matrix(df, figsize = (18,18))
plt.show()
