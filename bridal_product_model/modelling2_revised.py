import sys
import pandas as pd
import numpy as np
import random
import uuid
import pickle
import statsmodels.api as stat
import os

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
		## export train and test csv here
		train_raw = self.raw_data.loc[self.training_data_id,]
		test_raw = self.raw_data.loc[self.testing_data_id,]
		train_raw.to_csv('train.csv')
		test_raw.to_csv('test.csv')
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

	def data_classifier(self):
		for X,Y in zip(self.training_set_X,self.training_set_Y):
			carat = str(self.round_off_1d(X[0]))
			clarity = X[1]
			color = X[2]
			cut = X[3]
			shape = X[4]
			string = carat + ',' + clarity + ',' + color + ',' + cut + ',' + shape
			if string not in self.classes:
				self.classes.append(string)
			if string not in self.classified_data_dict_X.keys():
				self.classified_data_dict_X[string] = []
				self.classified_data_dict_X[string].append(X[0])
				self.classified_data_dict_Y[string] = []
				self.classified_data_dict_Y[string].append(Y)
			elif string in self.classified_data_dict_X.keys():
				self.classified_data_dict_X[string].append(X[0])
				self.classified_data_dict_Y[string].append(Y)
		for cata in self.classified_data_dict_X.keys():
			X_length = len(self.classified_data_dict_X[cata])
			Y_length = len(self.classified_data_dict_Y[cata])
			if X_length != Y_length:
				print "Warning, data error"
				break

	def classified_model(self):
		all_key = self.classified_data_dict_X.keys()
		l = len(all_key)
		UUID_list = []
		while len(UUID_list) < l:
			for i in range(l):
				string = str(uuid.uuid4())[:8]
				if string not in UUID_list:
					UUID_list.append(string)
		uuid_to_key = {}
		key_to_uuid = {}
		for key, name in zip(UUID_list, all_key):
			uuid_to_key[key] = name
			key_to_uuid[name] = key
		uuid_to_key_output = open('uuid_to_key.pickle', 'wb')
		pickle.dump(uuid_to_key, uuid_to_key_output)
		uuid_to_key_output.close()
		key_to_uuid_output = open('key_to_uuid.pickle', 'wb')
		pickle.dump(key_to_uuid, key_to_uuid_output)
		key_to_uuid_output.close()
		for key in all_key:
			print key
			X_data = self.classified_data_dict_X[key]
			Y_data = self.classified_data_dict_Y[key]
			if len(X_data) < 2:
				print "Lack of data, OLS regression skipped"
			else:
				model = stat.OLS(Y_data, X_data)
				result = model.fit()
				self.models[key] = result
				path = 'model/'
				if not os.path.exists(path):
					os.makedirs(path)
				model_name = 'model/' + key_to_uuid[key] + '.pickle'
				result.save(model_name)
				print "model formed and saved"
		
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
		print 'Prediction    Actual Price'
		for price, pred in zip(self.testing_set_Y, self.prediction):
			if pred == 'N/A':
				print '%s        %s' % (pred, price)
			else:
				print '%.2f        %s' % (pred, price)

	def training_set_RMSE(self):
		SSE = 0
		n = 0 
		for key in self.models.keys():
			array = self.models[key].resid
			n += len(array)
			for element in array:
				SSE += (element ** 2)
		RMSE = np.power(SSE/n, float(1)/2)
		print 'training set RMSE = %.2f' % (RMSE)

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

	def average_price(self):
		Y = self.training_set_Y + self.testing_set_Y
		print 'Average price = %.2f' % (np.mean(Y))

	def main(self):
		self.data_transform()
		self.data_dict_to_df()
		self.variable_exclude()
		self.data_random_sampler()
		self.check_data()
		self.data_classifier()
		self.classified_model()
		#self.run_prediction()
		#self.average_price()
		#self.training_set_RMSE()
		#self.testing_set_RMSE()
		#print self.training_set_X

model = jewellery_pricing_model(input1, input2)
model.main()