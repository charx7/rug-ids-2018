print('Im Working!')
# Package Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

# Custom Imports
from utils.preprocessor import preprocess
from utils.scaler import scale
from utils.class_balancer import doUpsamling
from utils.preprocessor import reduceDimentions

# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function
df = preprocess(rawData, labels, True)
###### Check-out our data ########
#print("The appended resulting dataframe is: \n", df)

# Scale the Dataset
# Drop the las column Dont need scailing on the labels XD
df = df.drop(['cancer'], axis=1)
df = scale(df)

# Get back the cancer label
df = preprocess(df, labels, False)

# Get the columns for future use
rows, columns = df.shape

df_values = df.values
# split to train and test data
X=df_values[:,0:columns - 1]
y=df_values[:,[columns - 1]].flatten()

# Split before upsampling
X_train, X_final_test, y_train, y_final_test = train_test_split(X, y, test_size=0.30,
 	random_state=45)

y_train_labels = pd.DataFrame(data=y_train[:], columns=['cancer'])

# Convert X_train to a df to attach the labels shape gets the num of columns
newNames = []
for i in range(X_train.shape[1]):
	newNames.append('x' + str(i+1))

# Transform into DF again
X_train_dataframe = pd.DataFrame(data=X_train[:], columns=[newNames])

# Call the pre-processor to attach the labels of 'cancer' again
df = preprocess(X_train_dataframe, y_train_labels, False)

# Upsampling uses random stuff we plug the meaning of life into it
np.random.seed(42)
# Class balancing: upsampling minority class
df_upsampled_ordered = doUpsamling(df)
df_upsampled = df_upsampled_ordered.sample(frac=1).reset_index(drop=True)
df_values = df_upsampled.values

# scale training data using standar scaler
#scaler = preprocessing.StandardScaler().fit(X_train)
#X_train_scaled = scaler.transform(X_train)

# split to train and test data
X=df_values[:,0:columns - 1]
y=df_values[:,[columns - 1]].flatten()

# esta mal produce 1 columna menos
# Now split the upsampled data for the X-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
 	random_state=42)

# neighbor list
neighbors = list(range(1,25))

# list with cv test scores
cv_scores = []

print("==========================")
print("K-NN crossvalidation start")
for k in neighbors:

	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X_train, y_train, cv=20, scoring='accuracy')

	cv_scores.append(scores.mean())

print("K-NN crossvalidation end")
print("========================")
print("Test score means: \n", cv_scores)

MSE = [1 - x for x in cv_scores]

# determining best k only on test score misclassification
optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error of the validation sets')
plt.show()

# Do the kNN with the optimal parameter and check for performance of the test
knn = KNeighborsClassifier(n_neighbors=optimal_k)

# fitting the model
knn.fit(X_train, y_train)
# Disregard this is for bug-squasing
print('shape of the x train ', X_train.shape)
print('shape of the y train ', y_train.shape)
print('shape of the x test ', X_test.shape)
print('shape of the x final test ', X_final_test.shape)

# predict the response
pred = knn.predict(X_final_test)

# evaluate accuracy
print ('The final accuracy score of our model is: ',
	accuracy_score(y_final_test, pred))

# plot accuracy on the test set vs different k's
accuracy_on_test = []
for k in neighbors:
	# Do the kNN with the optimal parameter and check for performance of the test
	knn = KNeighborsClassifier(n_neighbors=k)

	# fitting the model
	knn.fit(X_train, y_train)

	# predict the response
	pred = knn.predict(X_final_test)

	acc = accuracy_score(y_final_test, pred)

	accuracy_on_test.append(acc)

print("Accuracies on the test with different k's is: \n", accuracy_on_test)

# plot misclassification error vs k
plt.plot(neighbors, accuracy_on_test)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy on the test set')
plt.show()

print('\n')
print("========================")
print("Start Dim reduction")
# Now we are going to do the dim reduction methods
# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

# Call the custom function
df = preprocess(rawData, labels, True)

# Scale the Dataset
# Drop the las column Dont need scailing on the labels XD
df = df.drop(['cancer'], axis=1)
df = scale(df)

# Get back the cancer label
df = preprocess(df, labels, False)

# Reduce dimensions on the dataset using ANOVA
sortedAnovaResults, significantValues, reducedDf = reduceDimentions(df,
    'ANOVA', 0.01, reduce = True)

# Attach the labels of cancer ...again... damn pandas methods
reducedDf = preprocess(reducedDf, labels, False)

print("End Dim reduction")
print("========================")
print('\n')

print("========================")
print("Start train/test Split")

# Get the columns for future use
rows, columns = reducedDf.shape

df_values = reducedDf.values
# split to train and test data
X=df_values[:,0:columns - 1]
y=df_values[:,[columns - 1]].flatten()

# print dimensions as a bug sprayer
print('The dimension of X is: ', X.shape)
print('The dimension of y is: ', y.shape)
# Split before upsampling
X_train_toCV, X_final_test, y_train_toCV, y_final_test = train_test_split(X, y, test_size=0.30,
 	random_state=45)

print("End train/test Split")
print("========================")
print('\n')

print("========================")
print("Start scaling")
# Get a pandas df from the split
y_train_toCV = pd.DataFrame(data=y_train_toCV[:], columns=['cancer'])

# Convert X_train to a df to attach the labels shape gets the num of columns
newNames = []
for i in range(X_train_toCV.shape[1]):
	newNames.append('x' + str(i+1))

# Transform into DF again
X_train_toCV_dataframe = pd.DataFrame(data=X_train_toCV[:], columns=[newNames])

# Call the pre-processor to attach the labels of 'cancer' again
df = preprocess(X_train_toCV_dataframe, y_train_toCV, False)

print("End scaling")
print("========================")
print('\n')

print("========================")
print("Start Up-Sampling")

# Upsampling uses random stuff, so we plug the meaning of life into it
np.random.seed(42)

# Class balancing: upsampling minority class
df_upsampled_ordered = doUpsamling(df)
df_upsampled = df_upsampled_ordered.sample(frac=1).reset_index(drop=True)
df_values = df_upsampled.values

print("End Up-Sampling")
print("========================")
print('\n')


print("========================")
print("Start train-test split on the up-sampled & dim reduced dataset")

# split to train and test data
X=df_values[:,0:columns - 1]
y=df_values[:,[columns - 1]].flatten()

# For bug squasing
print('Dimensions of the vectos to split...')
print('Dimension of X is: ', X.shape)
print('Dimension of y is: ', y.shape)

# esta mal produce 1 columna menos
# Now split the upsampled data for the X-validation
X_xv_train, X_xv_test, y_xv_train, y_xv_test = train_test_split(X, y, test_size=0.30,
 	random_state=42)

print("End train-test split on the up-sampled & dim reduced dataset")
print("========================")
print('\n')

print("Start the X-val for hyper-param adjust")
print("========================")

# neighbor list
neighbors = list(range(1,25))

# list with cv test scores
cv_scores = []

for k in neighbors:

	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X_xv_train, y_xv_train, cv=10, scoring='accuracy')

	cv_scores.append(scores.mean())

print("Test score means: \n", cv_scores)

MSE = [1 - x for x in cv_scores]

# determining best k only on test score misclassification
optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error of the validation sets')
plt.show()


print("End the X-val for hyper-param adjust")
print("========================")
print('\n')

print("========================")
print("Start Model Evaluation")

# Do the kNN with the optimal parameter and check for performance of the test
knn = KNeighborsClassifier(n_neighbors=optimal_k)

# fitting the model
knn.fit(X_xv_train, y_xv_train)

# Disregard this is for bug-squasing
print('shape of the x train ', X_train.shape)
print('shape of the y train ', y_train.shape)
print('shape of the x test ', X_test.shape)
print('shape of the x final test ', X_final_test.shape)

# Predict the response using the data from the first split we did before
# upsampling
pred = knn.predict(X_final_test)

# evaluate accuracy
print ('The final accuracy score of our model is: ',
	accuracy_score(y_final_test, pred))

# plot accuracy on the test set vs different k's
accuracy_on_test = []
for k in neighbors:
	# Do the kNN with the optimal parameter and check for performance of the test
	knn = KNeighborsClassifier(n_neighbors=k)

	# fitting the model
	knn.fit(X_xv_train, y_xv_train)

	# predict the response
	pred = knn.predict(X_final_test)

	acc = accuracy_score(y_final_test, pred)

	accuracy_on_test.append(acc)

print("Accuracies on the test with different k's is: \n", accuracy_on_test)

# plot misclassification error vs k
plt.plot(neighbors, accuracy_on_test)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy on the test set')
plt.show()

print("End Model Evaluation")
print("========================")
