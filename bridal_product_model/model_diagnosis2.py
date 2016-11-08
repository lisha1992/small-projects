import math
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from statsmodels.regression.linear_model import OLSResults

class model_diagnosis:
	def __init__(self, test_file_name):
		self.test_file_name = test_file_name
		self.allModelName = self.getAllModelFileName()
		self.models = {}
		self.testing_set = pd.read_csv(self.test_file_name)
		self.testing_set_X = []
		self.testing_set_Y = []
		self.prediction = []
		self.headers = list(self.testing_set.columns.values)

	def getAllModelFileName(self):
		mypath = 'model/'
		return [f for f in listdir(mypath) if isfile(join(mypath, f))]

	def unpickle(self):
		for name in self.allModelName:
			key = name[:-7]
			model = OLSResults.load('model/' + name)
			self.models[key] = model

	def data_transform(self):
		# [carat, clarity, color, cut, 1]
		l = len(self.testing_set)
		for i in range(l):
			carat = self.testing_set.iloc[i]['carat']
			clarity = self.testing_set.iloc[i]['clarity']
			color = self.testing_set.iloc[i]['color']
			cut = self.testing_set.iloc[i]['cut']
			price = self.testing_set.iloc[i]['price']
			data_X = [carat,clarity,color,cut,1]
			self.testing_set_X.append(data_X)
			self.testing_set_Y.append(price)

	def round_off_1d(self, number):
		number = int(number*10)
		number = number - number%2
		return float(number)/10

	def run_prediction(self):
		for X in self.testing_set_X:
			carat = str(self.round_off_1d(X[0]))
			carat_value = float(X[0])
			clarity = X[1]
			color = X[2]
			cut = X[3]
			string = carat + ',' + clarity + ',' + color + ',' + cut
			if string in self.models.keys():
				prediction = self.models[string].predict(carat_value)
			else:
				prediction = 'N/A'
			self.prediction.append(prediction)
		'''
		print 'Prediction    Actual Price'
		for price, pred in zip(self.testing_set_Y, self.prediction):
			if pred == 'N/A':
				print '%s        %s' % (pred, price)
			else:
				print '%.2f        %s' % (pred, price)
		'''

	def testing_set_RMSE(self):
		total = float(0)
		count = 0
		for price, pred in zip(self.testing_set_Y, self.prediction):
			if not isinstance(pred, str):
				total += ((float(pred) - price) ** 2)
				count += 1
		mean = float(total)/count
		RMSE = np.power(mean, 0.5)
		print 'testing set RMSE = %.2f' % (RMSE)

	def main(self):
		self.unpickle()
		self.data_transform()
		self.run_prediction()
		self.testing_set_RMSE()

DIAG = model_diagnosis('test.csv')
DIAG.main()