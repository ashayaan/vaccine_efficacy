import pandas as pd
import numpy as np
import numpy.random as rand
from sklearn.linear_model import LogisticRegression
import sys


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
	return lr.score(test_data, test_labels)


if __name__ == "__main__":
	train_data_name = sys.argv[1]
	test_data_name = sys.argv[2]
	(train_data,train_labels) = trainingData(train_data_name)
	(test_data,test_labels) = testingData(test_data_name)
	lr = train(train_data, train_labels)
	print test(test_data, test_labels, lr)
		

