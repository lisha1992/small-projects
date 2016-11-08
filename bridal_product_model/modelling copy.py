'''
This is the regression model without any variable selection
20160505    Selina Ng
modified from Louis' script jewellery_pricing_model2.py
Model Form:
np.sqrt(price) ~ carat + carat:cut:color:clarity:shape

arguments: diamonds_synthetic.csv 0.8
changed by Louis (object serialization added)
'''

import sys
import pandas as pd
import numpy as np
import random
import statsmodels.api as stat
import statsmodels.formula.api as smf
import math

input1 = str(sys.argv[1])
input2 = float(sys.argv[2])

class jewellery_pricing_model:
	def __init__(self, file_name, training_set_size):
		self.file_name = file_name
		self.raw_data = pd.read_csv(self.file_name)
		self.headers = list(self.raw_data.columns.values)
		self.data_dict = {}
		self.classes = []
		self.classified_data_dict_X = {}
		self.classified_data_dict_Y = {}
		self.models = {}
		self.training_set_size = training_set_size
		self.training_set_X = []
		self.training_set_Y = []
		self.testing_set_X = []
		self.testing_set_Y = []
		self.prediction = []

	def df_to_list(self, df):
		array = []
		for num in df:
			array.append(num)
		return array

	def remove_nan(self, array):
		new_list = []
		for element in array:
			if np.isnan(element):
				new_list.append(int(0))
			else:
				new_list.append(int(element))
		return new_list
	
	def data_transform(self):
		# read the csv file transform into a dictionary
		print 'Reading the file: "' + self.file_name +'"'
		for colnames in self.headers:
			array = self.df_to_list(self.raw_data[colnames])
			# array = self.remove_nan(array)
			self.data_dict[colnames] = array
	
	def data_dict_to_df(self):
		index = self.data_dict.pop('id')
		self.data_id = index
		self.data_df = pd.DataFrame(self.data_dict, index=index)
		self.headers = list(self.data_df.columns.values)

	def variable_exclude(self):
		self.headers.remove('depth')
		self.headers.remove('table')
		self.headers.remove('x')
		self.headers.remove('y')
		self.headers.remove('z')

	def data_random_sampler2(self):
		print 'Training data sampling ' + str(self.training_set_size*100) +' %' + ' from whole dataset'
		rseq = random.sample(range(len(self.raw_data)),len(self.raw_data))
		brk  = int(round(len(rseq)*self.training_set_size))
		self.training_data_id = rseq[:brk]
		self.testing_data_id  = rseq[brk:]
	
	def round_off_1d(self, number):
		number = int(number*10)
		number = number - number%2
		return float(number)/10

	def string_breaker(self, string):
		l = len(string)
		num_list = []
		for i in range(l):
			if string[i] == ',':
				lis.append(i)
		a, b, c = num_list
		return float(string[:a]), string[a+1:b], string[b+1:c], string[c+1:]
		
	def classified_model2(self):
		key = 'np.sqrt(price) ~ carat + carat:clarity:color:cut:shape'
		print key
		ols_data = self.raw_data.loc[self.training_data_id,]
		if len(ols_data) < 2:
			print "Lack of data, OLS regression skipped"
		else:
			model = smf.ols(key, data=ols_data)
			result = model.fit()
			self.models[key] = result
			print "model formed"
			result.save("model.pickle")

	def main(self):
		self.data_transform()
		self.data_dict_to_df()
		self.variable_exclude()
		self.data_random_sampler2()
		self.classified_model2()

model = jewellery_pricing_model(input1, input2)
model.main()
print len(model.training_data_id)
print len(model.testing_data_id)


