import math
import pickle
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from statsmodels.regression.linear_model import OLSResults

class model_diagnosis:
	def __init__(self, string):
		self.string = string
		# format 'carat, clarity, color, cut, shape'
		self.allModelName = self.getAllModelFileName()
		self.models = {}
		self.prediction = []

	def getAllModelFileName(self):
		mypath = 'model/'
		return [f for f in listdir(mypath) if isfile(join(mypath, f))]

	def unpickle(self):
		pkl_file = open('uuid_to_key.pickle', 'rb')
		self.uuid_to_key = pickle.load(pkl_file)
		pkl_file.close()
		pkl_file = open('key_to_uuid.pickle', 'rb')
		self.key_to_uuid = pickle.load(pkl_file)
		pkl_file.close()
		for name in self.allModelName:
			key = name[:-7]
			key = self.uuid_to_key[key]
			model = OLSResults.load('model/' + name)
			self.models[key] = model

	def data_transform(self):
		# [carat, clarity, color, cut, shape, 1]
		string_list = self.string.split(',')
		string_list.append(1)
		return string_list

	def round_off_1d(self, number):
		number = int(number*10)
		number = number - number%2
		return float(number)/10

	def run_prediction(self):
		data_list = self.data_transform()
		carat = str(self.round_off_1d(data_list[0]))
		carat_value = float(data_list[0])
		clarity = data_list[1]
		color = data_list[2]
		cut = data_list[3]
		shape = data_list[4]
		string = carat + ',' + clarity + ',' + color + ',' + cut + ',' + shape
		if string in self.models.keys():
			prediction = self.models[string].predict(carat_value)
			print float(prediction)
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
		self.run_prediction()

DIAG = model_diagnosis('test.csv')
DIAG.main()