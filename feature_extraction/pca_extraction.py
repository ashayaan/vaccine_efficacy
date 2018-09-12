import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression

def PCAPrediction(df,test):
	pca = decomposition.PCA(n_components=51)
	pca.fit(df)
	x = pca.transform(df)

	c = list(df.label)
	lr = LogisticRegression()
	lr.fit(x,c)

	pca2 = decomposition.PCA(n_components=51)
	pca2.fit(test)
	test_data = pca2.transform(test)
	
	
	test_labels = test.label


	print 'shapr of train data ' + str(x.shape)
	print 'shape of test data ' + str(test_data.shape)
	print lr.score(test_data,test_labels)



	
if __name__ == '__main__':
	df = pd.read_csv('/home/shayaan/sem_9/RE/vaccine_efficacy/data_files/combined.csv')
	df = df.drop(columns='Unnamed: 0')

	df2 = pd.read_csv('/home/shayaan/sem_9/RE/vaccine_efficacy/data_files/day5/day5_noiseadded.csv')
	df2 = df2.drop(columns='Unnamed: 0')
	df2 = df2.dropna() 

	print df2.shape

	PCAPrediction(df,df2)
