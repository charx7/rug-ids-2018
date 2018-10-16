import pandas as pd
import numpy as np

def preprocess(rawData, labels):
    # Change the name of the columns of the training examples
    newNames = []
    for i in range(len(rawData.columns)):
        newNames.append('x' + str(i+1))

    # Change the column names
    rawData.columns=[newNames]
    #print("The basic structure of the data is: \n ", rawData)

    # Change the name of the column to "class"
    labels.columns = ['class']
    #print("The first labels are: \n ", labels)

    # Transform the labels into 0 for non diseased and 1 for AML positive
    labels['class'] = labels['class'].map({
        1: 0,
        2: 1
    })

    # Get values in an arrayform
    valuesToInsert = labels['class'].values
    # Slice the original data to get just the 179 labeled examples
    slicedRawData = rawData[:179]
    # Add the 'class' column based on the labels
    slicedRawData['class'] = valuesToInsert
    # Add the labeled columns to the numpy array
    appendedDf = slicedRawData

    return appendedDf
