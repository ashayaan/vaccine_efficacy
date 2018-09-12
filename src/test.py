import pandas as pd
import numpy as np
import numpy.random as rand
import itertools
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC,LinearSVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib
import sys


seq2 = pd.Series(np.arange(2))

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



class Vaccine(object):
	seq2 = pd.Series(np.arange(2))

	"""Vaccine efficacy class for training and testing various models"""
	def __init__(self, trainFile, testFile):
		self.trainFile = trainFile
		self.testFile = testFile
		self.__lr = LogisticRegression()
		self.__dtree = DecisionTreeClassifier()
		self.__rforest = RandomForestClassifier()
		self.__svm = LinearSVC(penalty='l2',random_state=0,loss='hinge',dual=True,fit_intercept=True,)
		self.__mlp = MLPClassifier(hidden_layer_sizes=(100, ),activation='relu',solver='adam',
								   batch_size='auto', learning_rate='constant', learning_rate_init=0.0001 )
		self.train_data = None
		self.train_labels = None
		self.test_data = None
		self.test_labels = None
		self.predicted_labels = None
		self.plot_data = None

	def trainingData(self):
		df = pd.read_csv(self.trainFile)
		df = df.drop(columns='Unnamed: 0')
		self.plot_data = df
		df = df.dropna()
		self.train_labels = df['label']
		self.train_data = df.drop(columns='label')

	def testingData(self):
		df = pd.read_csv(self.testFile)
		df = df.drop(columns='Unnamed: 0')
		df = df.dropna()
		self.test_labels = df['label']
		self.test_data = df.drop(columns='label')

	def data(self):
		self.trainingData()
		self.testingData()

	def plotData(self):
		df = self.plot_data
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

	@staticmethod
	def stringLabel(col):
		if col.label == 1:
			return 'Effective'
		else:
			return 'Not Effective'

	def pairwisePlot(self):
		# cols = ['1007_s_at', '1053_at', '117_at', '121_at','categories']
		df = self.plot_data
		cols = random.sample(df.columns,4)
		cols.append('categories')
		df['categories'] = df.apply(lambda col: self.stringLabel(col),axis=1)

		pp = sns.pairplot(df[cols],hue='categories')
		fig = pp.fig 
		fig.subplots_adjust(top=0.93, wspace=0.3)
		t = fig.suptitle('Genetic Attributes Pairwise Plots', fontsize=14)
		plt.show()

	def plotROC(self):
		false_positive_rate, true_positive_rate, thresholds = roc_curve(self.test_labels, self.predicted_labels)
		roc_auc = auc(false_positive_rate, true_positive_rate)
		plt.title('Receiver Operating Characteristic')
		plt.plot(false_positive_rate, true_positive_rate, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
		plt.legend(loc='lower right')
		plt.plot([0,1],[0,1],'b--')
		plt.xlim([-0.0,1.2])
		plt.ylim([-0.0,1.2])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()

	def trainLogisticRegression(self):
		self.__lr.fit(self.train_data,self.train_labels)	
	
	def testLogisticRegression(self):
		self.predicted_labels =  self.__lr.predict(self.test_data)
		print "Logistic Regression score " + str(self.__lr.score(self.test_data, self.test_labels))

	def trainDecesionTree(self):
		self.__dtree.fit(self.train_data,self.train_labels)

	def testDecesionTree(self):
		self.predicted_labels = self.__dtree.predict(self.test_data)
		print "Decision Tree Score " + str(self.__dtree.score(self.test_data,self.test_labels))
	
	def trainRandomForrest(self):
		self.__rforest.fit(self.train_data,self.train_labels)

	def testRandomForrest(self):
		self.predicted_labels = self.__rforest.predict(self.test_data)
		print "Random Forest Score " + str(self.__rforest.score(self.test_data,self.test_labels))

	def trainSVM(self):
		self.__svm.fit(self.train_data,self.train_labels)

	def testSVM(self):
		self.predicted_labels = self.__svm.predict(self.test_data)
		print "SVM score " + str(self.__svm.score(self.test_data,self.test_labels))

	def trainMLP(self):
		self.__mlp.fit(self.train_data,self.train_labels)

	def testMLP(self):
		self.predicted_labels = self.__mlp.predict(self.test_data)
		print "MLP score " + str(self.__mlp.score(self.test_data,self.test_labels))
 
if __name__ == "__main__":
	train_data_name = sys.argv[1]
	test_data_name = sys.argv[2]
	model = Vaccine(train_data_name,test_data_name)
	model.data()
	# model.plotData()
	model.pairwisePlot()

	# model.trainLogisticRegression()
	# model.testLogisticRegression()

	# plotConfusionMatrix(model.test_labels,model.predicted_labels)
	
	# model.trainDecesionTree()
	# model.testDecesionTree()

	# plotConfusionMatrix(model.test_labels,model.predicted_labels)
	
	# model.trainRandomForrest()
	# model.testRandomForrest()

	# plotConfusionMatrix(model.test_labels,model.predicted_labels)

	# model.trainSVM()
	# model.testSVM()
	
	# plotConfusionMatrix(model.test_labels,model.predicted_labels)

	# model.trainMLP()
	# model.testMLP()
	
	# plotConfusionMatrix(model.test_labels,model.predicted_labels)


	# model.plotROC()

	