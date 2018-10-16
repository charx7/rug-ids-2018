print('Im Working!')
# Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom Imports
from utils.preprocessor import preprocess
from utils.scaler import scale
from utils.dimentionallityReduction import doPCA

# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function
df = preprocess(rawData, labels)
###### Check-out our data ########
print("The appended resulting dataframe is: \n", df)

# Perform scailing on the data
scaledDf = scale(df)

# Perform PCA with 2 var on the scaledDf
pca_df, explainedVar = doPCA(scaledDf)
print('The pca is: \n', pca_df,
    '\n the culmutative sum of the variance explained is: ',
    explainedVar.cumsum())

# Plot our result for mad-viewz
fig = plt.figure()
# Plot config
plot = df.values
colors =  plot[:,186]

plt.scatter(pca_df['PCA 1'],pca_df['PCA 2'], c=colors, marker='.')
plt.title("2 dimensions PCA representation of the dataset", fontsize=15)
plt.xlabel('PCA_1', fontsize=13)
plt.ylabel('PCA_2', fontsize=13)
fig.set_size_inches(10, 10, forward=True)
plt.show()
