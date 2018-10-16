from sklearn.decomposition import PCA
import pandas as pd

# Fucntion that performs PCA TODO hardcoded to 2 increase to n
def doPCA(scaledDf, dimensionsToReduce):
    # Declare an instance of a PCA object
    pca = PCA(n_components= dimensionsToReduce)
    # Perform the fit of the pca
    pca_components = pca.fit_transform(scaledDf)

    # dataframe out of the resulting pca results
    columnLabels = []
    for i in range(dimensionsToReduce):
        columnLabels.append('PCA ' + str(i+1))

    pca_df = pd.DataFrame(data=pca_components, columns = columnLabels)
    # Get the explained variance
    explainedVariance = pca.explained_variance_ratio_
    # Get the culmutative sum
    # print ('The culmutative sum of the pca components is: \n',
    #     pca.explained_variance_ratio_.cumsum())

    return pca_df, explainedVariance
