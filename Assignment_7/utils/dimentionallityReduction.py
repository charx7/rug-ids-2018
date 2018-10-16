from sklearn.decomposition import PCA
import pandas as pd

# Fucntion that performs PCA TODO hardcoded to 2 increase to n
def doPCA(scaledDf):
    # Declare an instance of a PCA object
    pca = PCA(n_components=2)
    # Perform the fit of the pca
    pca_components_2 = pca.fit_transform(scaledDf)
    # dataframe out of the resulting pca results
    pca_df = pd.DataFrame(data=pca_components_2, columns = ['PCA 1', 'PCA 2'])
    # Get the explained variance
    explainedVariance = pca.explained_variance_ratio_
    # Get the culmutative sum
    # print ('The culmutative sum of the pca components is: \n',
    #     pca.explained_variance_ratio_.cumsum())

    return pca_df, explainedVariance
