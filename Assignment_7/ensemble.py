import numpy as np
import sklearn.tree as sk
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
from exploratory_analysis import reducedDf

class Classifier:
    def __init__(self, classifier):
        # Redefine class method
        self.classify = classifier

    def classify(self, data):
        # Given some training/test data, return a vector of zeros and ones
        # i.e. whether the patient has or hasnt the disease.
        prediction = np.zeros(1)
        return prediction

def majorityvote(data, classifiers):
    # Given data and a set of Classifier's, return the most predicted predictions
    predictions = np.array([c.classify(data) for c in classifiers])

    # Assuming the values are in columns
    return mode(predictions.transpose())[0]

	
	
## Decision Tree
from decisiontree import y_train
classifier_dt = sk.DecisionTreeClassifier(max_depth=20, min_samples_split= 93, max_leaf_nodes=43, min_samples_leaf=2, random_state=25)
dtModel = classifier_dt.fit(reducedDf.iloc[:, :-1], y_train)
decisionTree = Classifier(classifier_dt.predict)

## Decision Tree with different parameters
classifier_dt2 = sk.DecisionTreeClassifier(max_depth=10, min_samples_split= 50, max_leaf_nodes=30, min_samples_leaf=2, random_state=40)
dtModel2 = classifier_dt2.fit(reducedDf.iloc[:, :-1], y_train)
decisionTree2 = Classifier(classifier_dt2.predict)



## KNN
from KNN_classification import optimal_k, X_train_scaled, y_train
classifier_knn = KNeighborsClassifier(n_neighbors=optimal_k)
knnModel = classifier_knn.fit(X_train_scaled, y_train)
knnClassifier = Classifier(classifier_knn.predict)

## KNN with different parameters
classifier_knn2 = KNeighborsClassifier(n_neighbors=(optimal_k/2))
knnModel2 = classifier_knn2.fit(X_train_scaled, y_train)
knnClassifier2 = Classifier(classifier_knn2.predict)



# Add differently initialised classifiers to the ensemble
# Add Knn and DT classifiers to the ensemble
# Try new initialisations for different experiments?
# Ensemble becomes a new classifier defined by a function using the created classifiers.
classifiers = [decisionTree, decisionTree2, knnClassifier, knnClassifier2]
ensemble = Classifier(lambda data : majorityvote(data, classifiers))