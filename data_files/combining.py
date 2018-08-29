import pandas as pd

def combineData(name):
	df = pd.DataFrame()
	for i in name:
		temp = pd.read_csv(i)
		df = df.append(temp)
	df = df.dropna()
	df.to_csv('combined.csv',index=False)


if __name__ == '__main__':
	l = []
	for i in range(1,5):
		name = 'day'+str(i)+'/day'+str(i)+'_noiseadded.csv'
		l.append(name)
	combineData(l)