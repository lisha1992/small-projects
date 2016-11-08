'''
This is the regression model without any variable selection

Model Form:
Price ~ Carat + Cutting + Color + Clarity + x + y + z + depth + table
'''

import sys
import pandas as pd
import numpy as np
import random
import statsmodels.api as stat

input1 = str(sys.argv[1])
input2 = float(sys.argv[2])

class jewellery_pricing_model:
	def __init__(self, file_name, training_set_size):
		self.file_name = file_name
		self.raw_data = pd.read_csv(self.file_name)
		self.headers = list(self.raw_data.columns.values)
		self.data_type = {}
		self.data_dict = {}
		self.data_df = None
		self.data_id = None
		self.training_set_size = training_set_size
		self.regressors_b4diag = []
		self.training_data_id = []
		self.testing_data_id = []
		self.training_set_X = []
		self.testing_set_X = []
		self.training_set_Y = []
		self.testing_set_Y = []
		self.prediction = []
		self.result = None

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

	def check_data_type(self):
		keys = self.data_dict.keys()
		for key in keys:
			array = self.data_dict[key]
			check_array = []
			for num in array:
				if num not in check_array:
					check_array.append(num)
			if len(check_array) > 10:
				self.data_type[key] = 'continuous'
			else:
				self.data_type[key] = 'catagorial'

	def catagorial_data_transform(self):
		regressors = [i for i in self.headers if self.data_type[i] == 'catagorial']
		print regressors
		for X in regressors:
			all_var = []
			for num in self.data_dict[X]:
				if num not in all_var:
					all_var.append(num)
			length = len(all_var)
			if length > 2:
				count = 0
				for cat in all_var:
					if cat != 0:
						string = X + 'Is' + str(cat)
						num_list = []
						for num in self.data_dict[X]:
							if num == cat:
								num_list.append(1)
							else:
								num_list.append(0)
						if count < length-1:
							self.data_dict[string] = num_list
					count += 1
				self.data_dict.pop(X)
		self.headers = self.data_dict.keys()

	def adjust_carat_order(self):
		string = 'carat_ord2'
		array = []
		for carat in self.data_dict['carat']:
			array.append(float(carat)**2)
		self.data_dict[string] = array
		string = 'carat_ord3'
		array = []
		for carat in self.data_dict['carat']:
			array.append(float(carat)**3)
		self.data_dict[string] = array
		string = 'carat_ord4'
		array = []
		for carat in self.data_dict['carat']:
			array.append(float(carat)**4)
		self.data_dict[string] = array
		string = 'carat_ord5'
		array = []
		for carat in self.data_dict['carat']:
			array.append(float(carat)**5)
		self.data_dict[string] = array
		string = 'carat_ord6'
		array = []
		for carat in self.data_dict['carat']:
			array.append(float(carat)**6)
		self.data_dict[string] = array
		string = 'carat_ord7'
		array = []
		for carat in self.data_dict['carat']:
			array.append(float(carat)**7)
		self.data_dict[string] = array
		string = 'carat_ord8'
		array = []
		for carat in self.data_dict['carat']:
			array.append(float(carat)**8)
		self.data_dict[string] = array
		string = 'carat_ord9'
		array = []
		for carat in self.data_dict['carat']:
			array.append(float(carat)**9)
		self.data_dict[string] = array


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

	def data_random_sampler(self):
		print 'Training data sampling ' + str(self.training_set_size*100) +' %' + ' from whole dataset'
		all_data_frame = self.data_df
		data_id = self.data_id
		l = len(self.data_id)
		training_data_id = []
		testing_data_id = []
		while (float(len(training_data_id))/l) < self.training_set_size:
			select = random.choice(data_id)
			data_id.remove(select)
			training_data_id.append(select)
		self.data_id = training_data_id + data_id
		for num in self.data_id:
			if num not in training_data_id:
				testing_data_id.append(num)
		self.training_data_id = training_data_id
		self.testing_data_id = testing_data_id
		for ids in self.training_data_id:
			data_X = []
			data_y = 0
			for colname in self.headers:
				if colname != 'price':
					data_X.append(self.data_df.loc[ids,colname])
				else:
					data_y = self.data_df.loc[ids,colname]
			data_X.append(1)
			self.training_set_X.append(data_X)
			self.training_set_Y.append(data_y)
		for ids in self.testing_data_id:
			data_X = []
			data_y = 0
			for colname in self.headers:
				if colname != 'price':
					data_X.append(self.data_df.loc[ids,colname])
				else:
					data_y = self.data_df.loc[ids,colname]
			data_X.append(1)
			self.testing_set_X.append(data_X)
			self.testing_set_Y.append(data_y)

	def check_data(self):
		print 'Checking data status ...'
		warning = 'null'
		l = len(self.training_set_X[0])
		all_data = self.training_set_X + self.testing_set_X
		for data in all_data:
			if len(data) != l:
				print "Warning !! Data dimemsion not match !!"
				warning = 'error'
				break
		if warning == 'null':
			print "All data are matched in dimension"

	def OLS_regression(self):
		print 'Data are processed by OLS regression model ...'
		model = stat.OLS(self.training_set_Y, self.training_set_X)
		result = model.fit()
		self.result = result
		print result.summary()
		print self.headers
		print self.result.tvalues
		self.prediction = result.predict(self.testing_set_X)
		print 'Prediction    Actual Price'
		for price, pred in zip(self.testing_set_Y, self.prediction):
			print '%.2f        %s' % (pred, price)

	def training_set_RMSE(self):
		SSE = 0
		array = self.result.resid
		n = len(array)
		for element in array:
			SSE += (element ** 2)
		RMSE = np.power(SSE/n, float(1)/2)
		print 'training set RMSE = %.2f' % (RMSE)
		
	def testing_set_RMSE(self):
		total = float(0)
		count = 0
		for price, pred in zip(self.testing_set_Y, self.prediction):
			total += ((float(pred) - price) ** 2)
			count += 1
		mean = float(total)/count
		RMSE = np.power(mean, 0.5)
		print 'testing set RMSE = %.2f' % (RMSE)

	def average_price(self):
		Y = self.training_set_Y + self.testing_set_Y
		print 'Average price = %.2f' % (np.mean(Y))

	def main(self):
		self.data_transform()
		print self.file_name + ' is read'
		self.check_data_type()
		self.catagorial_data_transform()
		#self.adjust_carat_order()
		self.data_dict_to_df()
		# self.variable_exclude()
		print self.headers
		self.data_random_sampler()
		self.check_data()
		self.OLS_regression()
		self.average_price()
		self.training_set_RMSE()
		self.testing_set_RMSE()

model = jewellery_pricing_model(input1, input2)
model.main()
