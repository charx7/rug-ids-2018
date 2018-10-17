print('Im Working!')
# Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pprint

# Custom Imports
from utils.preprocessor import preprocess
from utils.scaler import scale
from utils.dimentionallityReduction import doPCA
from utils.dimentionallityReduction import doTSNE
from utils.dimentionallityReduction import doANOVA

# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function, True if you need to slice the dataset
df = preprocess(rawData, labels, True)
###### Check-out our data ########
print("The appended resulting dataframe is: \n", df)

# Perform scailing on the data
scaledDf = scale(df)

# Perform PCA with 2 var on the scaledDf
# Count the number of columns
rows, columns = scaledDf.shape
print('The number of columns is: ',columns)

# Remove the last column which contains the label results from the scaled df
scaledDf = scaledDf.drop(['x'+ str(columns)], axis=1)
print("The scaled resulting dataframe is: \n", scaledDf)

pca_df, explainedVar = doPCA(scaledDf, 20)
print('The pca is: \n', pca_df,
    '\n the culmutative sum of the variance explained is: ',
    explainedVar.cumsum())

# Plot our result for mad-viewz
# Plot config to get the color labels for the scatter plots
plot = df.values
colors =  plot[:,186]

fig = plt.figure()
plt.scatter(pca_df['PCA 1'],pca_df['PCA 2'], c=colors, marker='.')
plt.title("2 dimensions PCA representation of the dataset", fontsize=15)
plt.xlabel('PCA_1', fontsize=13)
plt.ylabel('PCA_2', fontsize=13)
fig.set_size_inches(10, 10, forward=True)
plt.show()

# Set label names for the plot
x_cords = []
for i in range(len(explainedVar[:])):
    x_cords.append(i + 1)

# plot the PCA scree Plot
fig, ax = plt.subplots()
ax.plot(x_cords, explainedVar, 'o-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

# plot the PCA culmutative variance explained Plot
fig, ax = plt.subplots()
ax.plot(x_cords, explainedVar.cumsum(), 'o-')
plt.title('Culmutative Variance Plot')
plt.xlabel('Principal Component')
plt.ylabel("`%` of the Var explained")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

# Do TSNE
tsne_results = doTSNE(scaledDf)
# Save the resulting columns of t-SNE for a plot
x_axis = tsne_results[:,0]
y_axis = tsne_results[:,1]

# Plot the results of the t-SNE
plt.scatter(x_axis, y_axis, c=colors, marker='.')
plt.title("2 dimensions t-SNE representation of the dataset", fontsize=15)
plt.xlabel('x-tsne', fontsize=13)
plt.ylabel('y-tsne', fontsize=13)
fig.set_size_inches(10, 10, forward=True)
plt.show()

# Moar dimentionallity Reduction via ANOVA
# Attach labels to the scaled dataset again...
scaledDf = preprocess(scaledDf, labels, False)
# Call the Anova function

# Empty array to save our resultz
anovaResults = []
for i in range(columns-1):
    # Save the column index we are working on
    currentIteration = "x" + str(i+1)
    # Call the custom ANOVA function
    currentF, currentP = doANOVA(scaledDf, currentIteration)

    # Save our results on a dictionary format
    currentIndex = i
    curentResults = {
        "currentIndex": currentIndex,
        "fStat": currentF,
        "pValue": currentP
    }
    # Append our results
    anovaResults.append(curentResults)

# Desc Sort the listed dictionary based on the P-value
sortedAnova = sorted(anovaResults, key=lambda k: k['pValue'])
# Pretty print to the console our results
pprint.pprint(sortedAnova)

pValuesArray = []
indicesArray = []
#print((sortedAnova[0])['pValue'])
# Turn the results into an array for a nice Viz
for i in range(len(sortedAnova)):
    pValuesArray.append(sortedAnova[i].get('pValue'))
    indicesArray.append(sortedAnova[i].get('currentIndex'))
print('The best indices according to ANOVA: ',indicesArray)
print('Their respective p-values are: ',pValuesArray)

#Plot for fun and profit

# Get just the p-values which are < 0.05 our CUTPOINT
# for statistical significance
significantValues = list(filter(lambda x: x < 0.01, pValuesArray))

print("We have: ", len(significantValues[:]),
    ' significant dimensions using a cutpoint of p-value <0.01 according to ANOVA')

x_cords = []
for i in range(len(significantValues[:])):
    x_cords.append(i + 1)

fig, ax = plt.subplots()
ax.plot(x_cords, significantValues, 'o-')
plt.title('P-Values plot')
plt.xlabel('Dimenension')
plt.ylabel("p-value")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()
