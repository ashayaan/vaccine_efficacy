import pandas as pd
import numpy as np
import numpy.random as rand
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys

seq2 = pd.Series(np.arange(2))


def trainingData(name):
	df = pd.read_csv(name)
	df = df.drop(columns='Unnamed: 0')
	train_labels = df['label']
	train_data = df.drop(columns='label')
	return train_data,train_labels

def testingData(name):
	df = pd.read_csv(name)
	df = df.drop(columns='Unnamed: 0')
	df = df.dropna()
	test_labels = df['label']
	test_data = df.drop(columns='label')

	return test_data,test_labels

def train(train_data, train_labels):
	lr = LogisticRegression()
	lr.fit(train_data,train_labels)	
	return lr
	
def test(test_data, test_labels, lr):
	predicted_labels =  lr.predict(test_data)
	return lr.score(test_data, test_labels),predicted_labels

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')
	print(cm)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def plotConfusionMatrix(test_labels,predicted_labels):
	cnf_matrix = confusion_matrix(test_labels, predicted_labels)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=seq2, normalize=True,title='Normalized confusion matrix')
	plt.show()


if __name__ == "__main__":
	train_data_name = sys.argv[1]
	test_data_name = sys.argv[2]
	(train_data,train_labels) = trainingData(train_data_name)
	(test_data,test_labels) = testingData(test_data_name)
	lr = train(train_data, train_labels)
	(score,predicted_labels) = test(test_data, test_labels, lr)
	print score
	plotConfusionMatrix(test_labels,predicted_labels)
		

