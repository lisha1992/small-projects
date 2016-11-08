import pandas as pd
import random
import statsmodels.api as sm

# start_date,was_viewed,was_bought,IsTradingDay,HSI_close,HIS_volume,IsHKHoilday,IsMainlandHoilday,XAU,high_temp,mean_temp,low_temp,Humid,Rain,Rain_Warning,Typhoon,CCI

class demand_prediction:
	def __init__(self, ratio):
		self.data_directory = 'data/all_combined.csv'
		self.data_dict = pd.read_csv(self.data_directory).to_dict('list')
		self.ratio = ratio
		self.nrow = 0
		self.training_X = list()
		self.training_viewedY = list()
		self.training_boughtY = list()
		self.testing_X = list()
		self.testing_viewedY = list()
		self.testing_boughtY = list()
		self.variable_selection = False

	def init_catagorial_variable(self):
		IsHK_nextDayHoilday = list()
		IsMainland_nextDayHoilday = list()
		l = len(self.data_dict['IsHKHoilday']) - 1
		for i in range(l):
			if self.data_dict['IsHKHoilday'][i+1] > 0:
				IsHK_nextDayHoilday.append(1)
			else:
				IsHK_nextDayHoilday.append(0)
			if self.data_dict['IsMainlandHoilday'][i+1] > 0:
				IsMainland_nextDayHoilday.append(1)
			else:
				IsMainland_nextDayHoilday.append(0)
		self.data_dict['IsHK_nextDayHoilday'] = IsHK_nextDayHoilday
		self.data_dict['IsMainland_nextDayHoilday'] = IsMainland_nextDayHoilday

	def trim_data(self):
		min_len = 0
		for key in self.data_dict.keys():
			l = len(self.data_dict[key])
			if min_len == 0 or l < min_len:
				min_len = l
		# print min_len
		for key in self.data_dict.keys():
			while len(self.data_dict[key]) > min_len:
				# print key
				self.data_dict[key].pop()
		all_length = 0
		for key in self.data_dict.keys():
			if all_length == 0:
				all_length = len(self.data_dict[key])
			elif all_length != len(self.data_dict[key]):
				print 'Warning !!! the length of data are different.'
				break
		self.nrow = all_length
		self.data_dict.pop("start_date")

	def data_sampling(self, ratio):
		if not self.variable_selection:
			self.all_variables = ['cnst', 'IsTradingDay','HSI_close','HSI_volume','IsHKHoilday','IsMainlandHoilday','IsHK_nextDayHoilday','IsMainland_nextDayHoilday','XAU','high_temp','mean_temp','low_temp','Humid','Rain','Rain_Warning','CCI','was_viewed','was_bought']
			all_variables = self.all_variables
		else:
			all_variables = self.all_variables
		# print self.all_variables
		self.variable_list = list()
		for i in range(len(all_variables)-2):
			if all_variables[i] != 'cnst':
				array = [all_variables[i], 'X'+str(i)]
				self.variable_list.append(array)
		all_dataset = list()
		for i in range(self.nrow):
			array = list()
			for var in all_variables:
				if var != 'cnst':
					array.append(self.data_dict[var][i])
			all_dataset.append(array)
		testing_size = int((1-self.ratio)*self.nrow)
		all_testing_data = random.sample(all_dataset, testing_size)
		for data in all_testing_data:
			if data in all_dataset:
				all_dataset.remove(data)
		all_training_data = all_dataset
		self.training_boughtY = list()
		self.training_viewedY = list()
		self.testing_boughtY = list()
		self.testing_viewedY = list()
		for data in all_training_data:
			self.training_boughtY.append(data.pop())
			self.training_viewedY.append(data.pop())
		self.training_X = sm.add_constant(all_training_data)
		for data in all_testing_data:
			self.testing_boughtY.append(data.pop())
			self.testing_viewedY.append(data.pop())
		self.testing_X = sm.add_constant(all_testing_data)
		# print len(self.training_X)
		# print len(self.training_boughtY)
		# print len(self.training_viewedY)
		# print len(self.testing_X)
		# print len(self.testing_boughtY)
		# print len(self.testing_viewedY)

	def modelling(self):
		# viewed_model = sm.OLS(self.training_viewedY,self.training_X)
		bought_model = sm.OLS(self.training_boughtY,self.training_X)
		# self.viewed_results = viewed_model.fit()
		self.bought_results = bought_model.fit()
		self.variable_selection = True
		# all_variables = ['cnst'] + self.all_variables
		# pvalue_list = self.bought_results.pvalues.tolist() + [0,0]
		# for i, j in zip(all_variables,pvalue_list):
		# 	print i + ' ' + str(j)
		# self.bought_results

	def backward_selection(self):
		pvalues = self.bought_results.pvalues.tolist() + [0,0]
		l = len(pvalues)
		# print str(len(pvalues)) + ' ' + str(len(self.all_variables))
		maximum_p = 0
		exclusion = ''
		for i in range(l):
			if i > 0 and pvalues[i] > maximum_p and pvalues[i] > 0.05:
				maximum_p = pvalues[i]
				exclusion = self.all_variables[i]
		if exclusion == '':
			return False
		else:
			print 'Variable (' + exclusion + ') excluded.'
			self.all_variables.remove(exclusion)
			return True

	def main(self):
		self.init_catagorial_variable()
		self.trim_data()
		self.data_sampling(self.ratio)
		self.modelling()
		while self.backward_selection():
			self.data_sampling(self.ratio)
			self.modelling()


model = demand_prediction(0.8)
model.main()
print model.bought_results.summary()

import numpy as np

RMSE = 0
for element in model.bought_results.resid:
	RMSE += element**2

mean = 0
for element in model.training_boughtY:
	mean += element

print 'MEAN: ' + str(mean/len(model.bought_results.resid))
print 'RMSE: ' + str(np.power(RMSE/len(model.bought_results.resid), 1.0/2))

for array in model.variable_list:
	print array




variable_dict = {
'X01' : 'IsTradingDay',
'X02' : 'HSI_close',
'X03' : 'HSI_volume',
'X04' : 'IsHKHoilday',
'X05' : 'IsMainlandHoilday',
'X06' : 'IsHK_nextDayHoilday',
'X07' : 'IsMainland_nextDayHoilday',
'X08' : 'XAU',
'X09' : 'high_temp',
'X10' : 'mean_temp',
'X11' : 'low_temp',
'X12' : 'Humid',
'X13' : 'Rain',
'X14' : 'Rain_Warning',
'X15' : 'CCI'
}
