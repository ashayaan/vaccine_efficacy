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
	def __init__(self, trainFile, testFile, feature_select):
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
		self.feature_select = feature_select
		self.use = None

	def featureSelect(self):
		if self.feature_select:
			return True
		else:
			return False

	def setFeature(self,use):
		self.use = use

	def trainingData(self):
		df = pd.read_csv(self.trainFile)
		df = df.drop(columns='Unnamed: 0')
		df = df.dropna()
		self.plot_data = df
		self.train_labels = df['label']
		
		#feature selection
		if self.use != None:
			df = df[self.use]
		
		self.train_data = df.drop(columns='label', errors='ignore')

	def testingData(self):
		df = pd.read_csv(self.testFile)
		df = df.drop(columns='Unnamed: 0')
		df = df.dropna()
		self.test_labels = df['label']
		
		#feature selection
		if self.use !=None:
			df = df[self.use]
		
		self.test_data = df.drop(columns='label', errors='ignore')

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
	model = Vaccine(train_data_name,test_data_name,True)

	if model.featureSelect():
		model.setFeature(['207083_s_at', '208754_s_at', '207574_s_at', '217993_s_at', '221888_at', '37424_at', '217290_at', '213264_at', '212451_at', '213934_s_at', '218642_s_at', '210324_at', '215359_x_at', '200956_s_at', '216910_at', '203344_s_at', '206862_at', '208455_at', '211599_x_at', '205093_at', '37028_at', '205906_at', '213528_at', '207795_s_at', '211976_at', '201865_x_at', '216277_at', '205497_at', '200606_at', '201324_at', '200852_x_at', '216061_x_at', '204502_at', '205233_s_at', '216092_s_at', '206133_at', '215226_at', '201726_at', '217903_at', '220838_at', '203125_x_at', '207289_at', '204009_s_at', '207657_x_at', '213748_at', '213756_s_at', '216901_s_at', '206929_s_at', '203380_x_at', '215492_x_at', '210146_x_at', '208009_s_at', '201711_x_at', '200603_at', '220584_at', '221750_at', '219421_at', '217062_at', '206841_at', '213749_at', '222283_at', '205334_at', '204616_at', '215110_at', '216923_at', '216429_at', '218320_s_at', '215623_x_at', '209591_s_at', '206787_at', '215036_at', '202802_at', '208199_s_at', '205482_x_at', '201758_at', '202490_at', '205366_s_at', '208660_at', '206944_at', '207663_x_at', '210051_at', '219484_at', '220664_at', '210482_x_at', '202614_at', '201932_at', '201843_s_at', '201606_s_at', '219352_at', '215192_at', '216036_x_at', '212832_s_at', '209999_x_at', '205787_x_at', '214358_at', '216471_x_at', '207268_x_at', '202114_at', '205247_at', '220746_s_at', '218658_s_at', '213946_s_at', '220512_at', '217221_x_at', '221224_s_at', '213498_at', '215521_at', '218832_x_at', '217113_at', '207567_at', '218337_at', '218160_at', '209222_s_at', '204181_s_at', '217057_s_at', '210427_x_at', '206569_at', '201966_at', '218351_at', '218464_s_at', '218565_at', '219860_at', '218416_s_at', '202300_at', '203915_at', '215835_at', '213363_at', '200002_at', '207466_at', '202851_at', '207275_s_at', '210340_s_at', '200967_at', '213503_x_at', '201535_at', '216808_at', '212067_s_at', '219089_s_at', '208079_s_at', '203015_s_at', '207067_s_at', '211084_x_at', '202628_s_at', '219609_at', '203801_at', '205972_at', '212875_s_at', '209106_at', '210054_at', '202432_at', '202046_s_at', '210517_s_at', '214759_at', '220261_s_at', '202377_at', '206682_at', '218875_s_at', '203947_at', '89977_at', '214963_at', '212602_at', '213766_x_at', '211430_s_at', '219508_at', '214970_s_at', '215225_s_at', '217134_at', '211674_x_at', '212823_s_at', '210669_at', '207707_s_at', '222289_at', '202239_at', '217240_at', '203754_s_at', '211006_s_at', '207011_s_at', '203629_s_at', '210678_s_at', '216200_at', '204752_x_at', '209359_x_at', '201981_at', '212684_at', '204779_s_at', '220017_x_at', '209480_at', '205545_x_at', '208098_at', '204530_s_at', '205267_at', '220638_s_at', '206606_at', '218188_s_at', '40359_at', '208351_s_at', '203678_at', '220960_x_at', '210741_at', '200828_s_at', '203609_s_at', '205091_x_at', '204896_s_at', '222034_at', '219254_at', '206094_x_at', '219536_s_at', '208438_s_at', '209725_at', '210219_at', '216265_x_at', '207460_at', '219944_at', '220322_at', '221369_at', '217890_s_at', '220119_at', '213144_at', '205161_s_at', '212804_s_at', '204387_x_at', '203772_at', '205159_at', '45653_at', '203277_at', '203975_s_at', '221524_s_at', '219500_at', '208695_s_at', '216266_s_at', '212336_at', '213517_at', '209740_s_at', '208950_s_at', '209264_s_at', '204460_s_at', '212140_at', '205313_at', '211327_x_at', '215004_s_at', '218216_x_at', '207149_at', '213496_at', '221177_at', '202180_s_at', '210799_at', '201428_at', '207424_at', '208425_s_at', '217403_s_at', '208565_at', '215204_at', '207253_s_at', '209954_x_at', '212969_x_at', '215365_at', '204207_s_at', '218769_s_at', '211300_s_at', '205095_s_at', '206562_s_at', '202218_s_at', '213375_s_at', '209799_at', '218749_s_at', '216892_at', '208789_at', '215766_at', '215277_at', '210407_at', '207126_x_at', '214780_s_at', '202824_s_at', '211718_at', '204578_at', '212944_at', '216934_at', '209953_s_at', '32099_at', '204339_s_at', '209734_at', '209782_s_at', '207247_s_at', '208196_x_at', '218181_s_at', '212833_at', '220951_s_at', '209706_at', '204285_s_at', '214925_s_at', '220081_x_at', '214862_x_at', '218972_at', '215109_at', '204784_s_at', '206686_at', '213371_at', '210839_s_at', '209612_s_at', '220272_at', '202646_s_at', '216976_s_at', '212151_at', '212608_s_at', '211989_at', '219164_s_at', '213485_s_at', '203509_at', '222367_at', '204359_at', '221014_s_at', '218433_at', '208909_at', '220004_at', '220505_at', '215087_at', '210378_s_at', '218732_at', '211582_x_at', '204631_at', '203022_at', '206065_s_at', '213209_at', '212928_at', '210132_at', '203159_at', '211127_x_at', '216213_at', '208856_x_at', '204846_at', '221568_s_at', '214443_at', '221248_s_at', '214048_at', '200815_s_at', '216332_at', '206736_x_at', '213141_at', '212722_s_at', '216562_at', '203979_at', '209796_s_at', '209225_x_at', '217051_s_at', '216879_at', '201250_s_at', '212805_at', '205017_s_at', '217505_at', '202098_s_at', '218826_at', '211852_s_at', '221472_at', '201655_s_at', '217973_at', '218503_at', '208169_s_at', '208943_s_at', '216584_at', '202874_s_at', '209288_s_at', '216813_at', '211401_s_at', '206618_at', '209685_s_at', '204026_s_at', '205310_at', '208728_s_at', '214380_at', '219785_s_at', '212919_at', '215129_at', '214379_at', '209458_x_at', '47571_at', '219945_at', '219797_at', '208986_at', '207012_at', '205956_x_at', '221517_s_at', '204011_at', '212366_at', '221551_x_at', '204195_s_at', '215309_at', '206045_s_at', '216959_x_at', '218461_at', '216607_s_at', '216874_at', '217599_s_at', '221559_s_at', '213875_x_at', '205647_at', '204385_at', '218375_at', '203718_at', '207915_at', '200931_s_at', '214616_at', '204235_s_at', '201191_at', '216597_at', '216177_at', '218192_at', '203215_s_at', '201697_s_at', '206829_x_at', '201430_s_at', '217108_at', '207442_at', '212667_at', '217832_at', '207290_at', '205289_at', '204157_s_at', '217503_at', '217842_at', '210078_s_at', '216985_s_at', '218771_at', '217128_s_at', '210533_at', '211122_s_at', '204757_s_at', '216776_at', '212153_at', '207376_at', '203994_s_at', '220764_at', '221433_at', '214974_x_at', '213965_s_at', '206468_s_at', '207877_s_at', '208447_s_at', '202776_at', '211795_s_at', '219547_at', '208677_s_at', '213466_at', '210995_s_at', '214005_at', '215317_at', '217152_at', '208577_at', '208608_s_at', '206694_at', '211919_s_at', '1053_at', '215209_at', '215144_at', '1438_at', '222274_at', '212994_at', '219689_at', '36920_at', '220680_at', '214439_x_at', '222309_at', '221083_at', '221591_s_at', '200753_x_at', '214805_at', '202719_s_at', '219404_at', '205337_at', '216696_s_at', '216598_s_at', '218971_s_at', '209127_s_at', '221349_at', '204028_s_at', '204016_at', '213678_at', '205103_at', '217495_x_at', '210225_x_at', '209439_s_at', '205266_at', '200622_x_at', '216744_at', '214507_s_at', '201529_s_at', '204161_s_at', '205550_s_at', '209932_s_at', '210528_at', '221102_s_at', '202872_at', '206723_s_at', '204687_at', '206197_at', '201724_s_at', '201745_at', '210890_x_at', '203886_s_at', '204607_at', '213575_at', '205104_at', '205883_at', '220655_at', '205440_s_at', '214245_at', '221541_at', '201310_s_at', '208556_at', '200007_at', '212241_at', '218645_at', '203168_at', '221990_at', '219465_at', '212084_at', '212612_at', '211978_x_at', '218644_at', '210416_s_at', '213732_at', '217266_at', '213804_at', '201936_s_at', '214743_at', '215970_at', '205846_at', '215050_x_at', '213080_x_at', '52975_at', '209845_at', '210133_at', '206190_at', '218031_s_at', '217156_at', '220879_at', '221915_s_at', '211565_at', '213188_s_at', '218742_at', '218792_s_at', '222298_at', '211964_at', '205520_at', '217473_x_at', '218275_at', '206288_at', '215711_s_at', '202191_s_at', '220728_at', '212434_at', '221982_x_at', '204648_at', '213551_x_at', '215833_s_at', '203198_at', '200961_at', '206161_s_at', '202691_at', '205569_at', '200703_at', '213806_at', '219759_at', '210306_at', '218615_s_at', '204456_s_at', '220184_at', '210039_s_at', '203752_s_at', '212250_at', '204742_s_at', '207391_s_at', '213159_at', '213534_s_at', '201521_s_at', '221959_at', '202987_at', '218544_s_at', '200833_s_at', '218038_at', '217528_at', '217546_at', '218250_s_at', '201856_s_at', '220647_s_at', '205707_at', '213416_at', '202065_s_at', '217667_at', '212520_s_at', '214343_s_at', '203515_s_at', '217345_at', '207138_at', '203284_s_at', '213916_at', '206505_at', '205263_at', '221767_x_at', '211198_s_at', '209851_at', '202077_at', '208730_x_at', '219438_at', '214250_at', '208763_s_at', '208694_at', '207569_at', '202742_s_at', '222355_at', '211284_s_at', '207561_s_at', '218486_at', '212211_at', '211576_s_at', '218685_s_at', '218517_at', '220932_at', '215107_s_at', '208648_at', '204471_at', '211111_at', '202714_s_at', '202198_s_at', '204711_at', '200759_x_at', '37547_at', '215342_s_at', '218699_at', '205287_s_at', '219196_at', '210014_x_at', '215999_at', '216222_s_at', '213653_at', '209767_s_at', '218902_at', '201269_s_at', '221513_s_at', '208423_s_at', '204573_at', '202695_s_at', '201005_at', '203044_at', '212349_at', '212764_at', '217820_s_at', '210278_s_at', '210185_at', '202946_s_at', '214065_s_at', '211870_s_at', '210333_at', '202454_s_at', '218812_s_at', '206220_s_at', '215496_at', '204868_at', '213939_s_at', '201512_s_at', '217366_at', '205967_at', '211738_x_at', '215788_at', '209497_s_at', '208725_at', '203846_at', '211931_s_at', '213839_at', '205325_at', '201245_s_at', '212834_at', '200726_at', '203076_s_at', '201654_s_at', '210233_at', '218465_at', '208980_s_at', '215658_at', '201773_at', '204663_at', '207726_at', '208087_s_at', '204593_s_at', '37796_at', '205498_at', '210317_s_at', '203891_s_at', '212840_at', '210267_at', '200830_at', '220875_at', '216168_at', '209446_s_at', '201889_at', '203318_s_at', '214496_x_at', '205081_at', '211107_s_at', '205288_at', '217746_s_at', '208753_s_at', '211118_x_at', '208415_x_at', '209908_s_at', '201466_s_at', '220490_at', '217866_at', '207358_x_at', '200762_at', '212751_at', '221111_at', '209166_s_at', '200885_at', '206601_s_at', '219513_s_at', '205768_s_at', '209530_at', '220705_s_at', '201126_s_at', '205351_at', '209454_s_at'])

	model.data()
	model.plotData()
	model.pairwisePlot()

	model.trainLogisticRegression()
	model.testLogisticRegression()

	# plotConfusionMatrix(model.test_labels,model.predicted_labels)
	
	# model.trainDecesionTree()
	# model.testDecesionTree()

	# plotConfusionMatrix(model.test_labels,model.predicted_labels)
	
	# model.trainRandomForrest()
	# model.testRandomForrest()

	plotConfusionMatrix(model.test_labels,model.predicted_labels)
	model.plotROC()


	model.trainSVM()
	model.testSVM()
	
	plotConfusionMatrix(model.test_labels,model.predicted_labels)

	# model.trainMLP()
	# model.testMLP()
	
	# plotConfusionMatrix(model.test_labels,model.predicted_labels)


	model.plotROC()

	