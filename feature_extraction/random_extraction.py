import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression


def sampleColumns(df):
	x = random.randint(1,1000)
	col = random.sample(df.columns,x)
	with open('colused.txt','a') as file:
		file.write(str(col))
		file.write('\n')

	print 'Column Sample Size ' + str(x)
	return col

def seperatData(df,cols):
	label = df.label
	data = df[cols]
	return data,label

def trainLogisticRegression(train_data,train_labels):
	lr = LogisticRegression()
	lr.fit(train_data,train_labels)
	return lr

def testLogisticRegression(lr,test_data,test_labels):
	return lr.score(test_data,test_labels)

if __name__ == '__main__':
	train = pd.read_csv('/home/shayaan/sem_9/RE/vaccine_efficacy/data_files/combined.csv')
	test = pd.read_csv('/home/shayaan/sem_9/RE/vaccine_efficacy/data_files/day5/day5_noiseadded.csv')
	train = train.dropna()
	test = test.dropna()
	test = test.drop(columns='Unnamed: 0')
	train = train.drop(columns='Unnamed: 0')

	for i in range(10):
		columns = sampleColumns(train)
		train_data,train_labels = seperatData(train,columns)
		test_data,test_labels = seperatData(test,columns)
		lr = trainLogisticRegression(train_data,train_labels)
		print 'Score ' + str(testLogisticRegression(lr,test_data,test_labels))
		print