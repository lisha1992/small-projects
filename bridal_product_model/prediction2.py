import sys
import math
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from statsmodels.regression.linear_model import OLSResults

input1 = str(sys.argv[1])

class model_diagnosis:
	def __init__(self, string):
		self.string = string
		# format 'carat, clarity, color, cut'
		self.allModelName = self.getAllModelFileName()
		self.models = {}
		self.prediction = []

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
		string_list = self.string.split(',')
		string_list.append(1)
		return string_list
		
	def round_off_1d(self, number):
		number = int(number*10)
		number = number - number%2
		return float(number)/10

	def run_prediction(self):
		data_list = self.data_transform()
		carat = str(self.round_off_1d(float(data_list[0])))
		carat_value = float(data_list[0])
		clarity = data_list[1]
		color = data_list[2]
		cut = data_list[3]
		string = carat + ',' + clarity + ',' + color + ',' + cut
		if string in self.models.keys():
			prediction = self.models[string].predict(carat_value)
			print float(prediction)
		else:
			print 'not enough data for model building'
			print 'no prediction will be generated'
		

	def main(self):
		self.unpickle()
		self.run_prediction()

DIAG = model_diagnosis(input1)
DIAG.main()