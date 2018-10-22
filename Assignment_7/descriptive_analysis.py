print('Im Working!')
# Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE
import time

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
pca_df, explainedVar = doPCA(scaledDf, 20)
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

# Now we are going to do t-TSNE
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(scaledDf.values)
print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
print(tsne_results)
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