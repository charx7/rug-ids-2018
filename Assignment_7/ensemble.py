import numpy as np
from scipy.stats import mode

class Classifier:
    def __init__(self, classifier):
        # Redefine class method
        self.classify = classifier

    def classify(self, data):
        # Given some training/test data, return a vector of zeros and ones
        # i.e. whether the patient has or hasnt the disease.
        prediction = np.zeros(1)
        return prediction

## E.g. decision tree
##   Define and train decision tree
classifier_dt = DecisionTreeClassifier(max_depth=20, min_samples_split= 93, max_leaf_nodes=43, min_samples_leaf=2, random_state=25)
dtModel = classifier_dt.fit(reducedDf.iloc[:, :-1], y_train)
##   Pass the classification function to the class
decisionTree = Classifier(classifier_dt.predict)


def majorityvote(data, classifiers):
    # Given data and a set of Classifier's, return the most predicted predictions
    predictions = np.array([c.classify(data) for c in classifiers])

    # Assuming the values are in columns
    return mode(predictions.transpose())[0]

# Add differently initialised classifiers to the ensemble
# Add Knn and DT classifiers to the ensemble
# Try new initialisations for different experiments?
# Ensemble becomes a new classifier defined by a function using the created classifiers.
classifiers = [decisionTree, decisionTree2, knnClassifier1, knnClassifier2]
ensemble = Classifier(lambda data : majorityvote(data, classifiers))
