from sklearn.utils import resample
import pandas as pd

def doUpsamling(df):
	# Separate majority and minority classes
	# Get a dataframe of TRUE of FALSE
	selection = df['cancer'] == 1
    # Select just the indices you want with the .loc function
	df_minority = df.loc[selection.values.flatten()]
	df_majority = df.loc[~selection.values.flatten()]
	# Upsample minority class
	# with replacement, to match the majority class and seed used to reproduce results
	df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=112358)
	
	print("Length of upsampled minority class: ", len(df_minority_upsampled))
	print("Length of majority class: ", len(df_majority))
	
	df_upsampled = pd.concat([df_minority_upsampled, df_majority])
	print("Length of upsampled dataset: ", len(df_upsampled))
	
	return df_upsampled