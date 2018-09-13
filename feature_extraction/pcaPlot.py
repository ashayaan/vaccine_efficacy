import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib
import sys
from sklearn import decomposition
import random



def readFeatures(name):
	with open(name,'r') as file:
		columns = file.read()

	columns = columns.lstrip('[')
	columns = columns.rstrip(']')
	columns = columns.replace("'","")
	columns = columns.replace(" ","")
	columns = columns.split(',')
	return columns

def stringLabel(col):
	if col.label == 1:
		return 'Effective'
	else:
		return 'Not Effective'

def pairwisePlot(df):
	cols = random.sample(df.columns,3)
	cols.append('categories')
	df['categories'] = df.apply(lambda col: stringLabel(col),axis=1)

	pp = sns.pairplot(df[cols],hue='categories')
	fig = pp.fig 
	fig.subplots_adjust(top=0.93, wspace=0.3)
	t = fig.suptitle('Genetic Attributes Pairwise Plots', fontsize=14)
	plt.show()

def pcaAnalysisn(df):
	pca = decomposition.PCA(n_components=200)
	pca.fit(df)
	x = pca.transform(df)
	return x

def plotData(df):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	xname,yname,zname = random.sample(df.columns,3)
	xs = df[xname]
	ys = df[yname]
	zs = df[zname]
	colors = ['red','blue']
	c = list(df.label)
	ax.scatter(xs, ys, zs, c=c, s=50, alpha=0.6, edgecolors='w',cmap=matplotlib.colors.ListedColormap(colors))
	ax.set_xlabel(xname)
	ax.set_ylabel(yname)
	ax.set_zlabel(zname)
	plt.show()

if __name__ == '__main__':
	cols = readFeatures('feature_taken.txt')
	cols.append('label')
	df = pd.read_csv('/home/shayaan/sem_9/RE/vaccine_efficacy/data_files/combined.csv')
	df = df[cols]
	
	pairwisePlot(df.copy())

	labels = np.asarray(df.label)

	df = df.drop(columns='label', errors='ignore')
	df = pcaAnalysisn(df)
	
	l = labels.reshape(df.shape[0],1)
	df = np.concatenate((df, l),axis=1)
	# print df.shape[1]

	cols = range(df.shape[1]-1)
	cols.append('label')
	data = pd.DataFrame(df, columns=cols)

	# plotData(data)

