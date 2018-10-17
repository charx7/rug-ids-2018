print('Im Working!')
# Package Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

# Custom Imports
from utils.preprocessor import preprocess
from utils.scaler import scale
from utils.class_balancer import doUpsamling

# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function
df = preprocess(rawData, labels, True)
###### Check-out our data ########
#print("The appended resulting dataframe is: \n", df)

# Class balancing: upsampling minority class
df_upsampled_ordered = doUpsamling(df)
df_upsampled = df_upsampled_ordered.sample(frac=1).reset_index(drop=True)
df_values = df_upsampled.values

# split to train and test data
X=df_values[:,0:186]
y=df_values[:,[186]].flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=112358)

# scale training data using standar scaler
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

# neighbor list
neighbors = list(range(1,25))
# list with cv test scores
cv_scores_test = []
# list with cv train scores
cv_scores_train = []
print("==========================")
print("K-NN crossvalidation start")
for k in neighbors:
	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_validate(knn, X_train_scaled, y_train, cv=10, scoring='accuracy', return_train_score=True)
	#print("For {} neighbors we have: {}".format(k,scores['test_score']))
	cv_scores_test.append(scores['test_score'].mean())
	cv_scores_train.append(scores['train_score'].mean())
print("K-NN crossvalidation end")
print("========================")
print("Test score means: \n", cv_scores_train)

# mock estimator - don't take it seriously
# changing to misclassification error
MSE_test = [1 - score for score in cv_scores_test]

# determining best k only on test score misclassification
optimal_k = neighbors[MSE_test.index(min(MSE_test))]
print("The optimal number of neighbors is %d" % optimal_k)

