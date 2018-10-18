from sklearn.tree import DecisionTreeClassifier, export_graphviz
from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV,
from sklearn.metrics import accuracy_score,confusion_matrix
from utils.class_balancer import doUpsamling
from utils.preprocessor import preprocess, reduceDimentions
from utils.scaler import scale
import pandas as pd
import numpy as np

# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

#set test to last 180 entries in the rawData set
realTest = rawData.tail(n=180)

#get first 179 records and label columns
df = preprocess(rawData, labels, True)

#remove last column before scaling
df = df.iloc[:, :-1]

#scale raw data
scaled_df = scale(df)

#append class labels again
prepDf = preprocess(scaled_df, labels, False)

#define relevant Features
#relevantFeatures = ['x39','x123','x130','x56','x157','x32'] #best dimensions??

#define train and target sets
train,test =train_test_split(prepDf, test_size=0.30, random_state = 45)
x_train = train.iloc[:, :-1]
x_test = test.iloc[:, :-1]
y_train = train['cancer']
y_test = test['cancer']

#oversample the minority class in x_train and y_train data
np.random.seed(42)
x_train, y_train = doUpsamling(x_train, y_train)

#again create labels and convert train sets into dataframes
newNames = []
for i in range(x_train.shape[1]):
	newNames.append('x' + str(i+1))

x_train = pd.DataFrame(data=x_train[:], columns=[newNames])
y_train = pd.DataFrame(data=y_train[:], columns=['cancer'])

#combine training sets before feature selection
combinedDf = preprocess(x_train, y_train, False)

#use ANOVA on training-only set to employ feature reduction
sortedAnovaResults, significantValues, reducedDf = reduceDimentions(combinedDf,
    'ANOVA', 0.01, reduce = True)

#append class labels again
reducedDf = reducedDf.join(combinedDf.iloc[:,-1])

#Set parameters for hyperparameter search with RandomizedSearchCV   "min_impurity_decrease": randint(0.005,0.5)
parameters = {"min_samples_split":randint(10,200),"max_depth":randint(3,50), "criterion":['gini'],"max_leaf_nodes": randint(2,200)}

classifier = DecisionTreeClassifier()
classifier_cv = RandomizedSearchCV (classifier,parameters, cv = 10, n_jobs = -1, n_iter=10000)

dtModel = classifier_cv.fit(reducedDf.iloc[:, :-1],y_train)

print("best parameters:{}".format(classifier_cv.best_params_))
print("best score is {}".format(classifier_cv.best_score_))

#store param values from dict into list
parameterValuesList = list(classifier_cv.best_params_.values())

#Build the dtModel with best parameters
classifier = DecisionTreeClassifier(max_depth=parameterValuesList[1], max_leaf_nodes= parameterValuesList[2], min_samples_split= parameterValuesList[3])

#fit into model again
dtModel = classifier.fit(reducedDf.iloc[:, :-1],y_train)

#return prediction array of 0 and 1
prediction = classifier.predict(x_test[reducedDf.iloc[:, :-1].columns])

#compute accuracy score
accuracyScore = accuracy_score(y_test,prediction)*100
print("Accuracy for current decision tree is ", round(accuracyScore,1), "%")

#create confusion matrix
cmatrix = confusion_matrix(y_test,prediction)

#create column names for realTest dataframe
realTest = rawData.tail(n=180)
newNames = []
for i in range(len(realTest.columns)):
    newNames.append('x' + str(i + 1))

#rename colors of realTest
realTest.columns = newNames

#filter out class column from column names of reducedDf
filteredColumnNames = [i[0] for i in list(reducedDf)]
del filteredColumnNames[-1]

#filter out irrelevant columns from realTest
realTest = realTest[filteredColumnNames]
#predict cancer patients
realPrediction = classifier.predict(realTest)
sum(realPrediction)

#output the decision tree
with open("output.dot", "w") as output_file:
    export_graphviz(classifier, out_file=output_file)
