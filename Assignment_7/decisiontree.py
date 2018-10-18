from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from class_balancer import doUpsamling
from utils.preprocessor import preprocess
from utils.scaler import scale
import pandas as pd


# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

#set realTest to last 180 entries in the rawData set
realTest = rawData.tail(n=180)

#obtain training data
#do preprocessing
df = preprocess(rawData, labels)
#balance classes with upsampling of minority class
df = doUpsamling(df)

#define decision tree classifier
classifier = DecisionTreeClassifier(min_samples_split= 100)

#define relevant Features
relevantFeatures = ['x39','x123','x130','x56','x157','x32'] #best dimensions??

#define train and target sets
train,test =train_test_split(df, test_size=0.30)
x_train = train[relevantFeatures]
x_test = test[relevantFeatures]
y_train = train['class']
y_test = test['class']

#run model
dtModel = classifier.fit(x_train,y_train)

#return prediction array of 0 and 1
prediction = classifier.predict(x_test)

#compute accuracy score
accuracyScore = accuracy_score(y_test,prediction)*100
print("Accuracy for current decision tree is ", round(accuracyScore,1), "%")

#cross validation score
cross_val_score(classifier, x_train, y_train, cv=5)

#classification report
target_names = ['class 0', 'class 1']
print(classification_report(y_test, prediction, target_names=target_names))

#confusion matrix
confusion_matrix(y_test,prediction)

#number of predicted cancer patients in the realTest set
newNames = []
for i in range(len(realTest.columns)):
    newNames.append('x' + str(i + 1))
realTest.columns = [newNames]
realTest = realTest[relevantFeatures]
realPrediction = classifier.predict(realTest)
sum(realPrediction)
print("The number of cancer patients are ", sum(realPrediction))
