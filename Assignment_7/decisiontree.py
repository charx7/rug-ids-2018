import imageio as imgo
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from class_balancer import doUpsamling
from utils.preprocessor import preprocess
from sklearn.decomposition import PCA
from utils.scaler import scale
from matplotlib import pyplot as plt
import graphviz
import pandas as pd
import numpy as np
import io
import pydotplus


# Read the data
rawData = pd.read_csv("data/data.csv", header=None)
labels = pd.read_csv("data/labels.csv", header=None)

#set test to last 180 entries in the rawData set
#test = rawData.tail(n=180)

#obtain training data
#do preprocessing
df = preprocess(rawData, labels)
#balance classes with upsampling of minority class
df = doUpsamling(df)
#train = df

####retrieve the features with highest variance based on PCA analysis
###scaledDf = scale(df)
###pca = PCA(n_components=2)
###pca.fit_transform(scaledDf)
###relFeat = pd.DataFrame(pca.components_,columns=scaledDf.columns,index = ['PCA 1','PCA 2'])
###
####sort columns by PCA-1 to find features with highest varience
###sorted_df = relFeat.sort_values(by = ['PCA 1'], axis =1)
####Last 20 features with highest PCA variance for PCA-1
###pca1 = sorted_df[sorted_df.columns[-20:]]
###
####sort columns by PCA-2 to find features with*highest varience
###sorted_df = relFeat.sort_values(by = ['PCA 2'], axis =1)
####Last 20 features with highest PCA variance for PCA-2
###pca2 = sorted_df[sorted_df.columns[-20:]]

#define decision tree classifier
classifier = DecisionTreeClassifier(min_samples_split= 100)
#define relevant Features
relevantFeatures = ['x39','x123','x130','x56','x157','x32'] #best dimensions??
#define train and target sets
train,test =train_test_split(df, test_size=0.15)
x_train = train[relevantFeatures]
x_test = test[relevantFeatures]
y_train = train['class']
y_test = test['class']

#run model
dtModel = classifier.fit(x_train,y_train)

###create function to output decision tree image
##def create_img(decisionTree,relevantFeatures, path):
##    file = io.StringIO()
##    export_graphviz(decisionTree, out_file=file ,feature_names=relevantFeatures)
##    pydotplus.graph_from_dot_data(file.getvalue()).write_png(path)
##    img = imgo.imread(path)
##    plt.rcParams['figure.figsize'] = (25,25)
##    plt.imshow(img)

#create_img(dtModel,relevantFeatures,'dt_01.png')

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
