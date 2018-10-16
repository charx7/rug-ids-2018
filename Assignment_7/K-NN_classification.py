print('Im Working!')
# Package Imports
import pandas as pd
import numpy as np

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
