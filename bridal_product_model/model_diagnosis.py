import math
import pandas as pd
from statsmodels.regression.linear_model import OLSResults

class model_diagnosis:
	def __init__(self, model_file_name, test_file_name):
		self.model_file_name = model_file_name
		self.test_file_name = test_file_name
		self.model = OLSResults.load(self.model_file_name)
		self.testing_set = pd.read_csv(self.test_file_name)
		self.prediction = []

	def testing_set_RMSE(self):
		err  = self.testing_set['price'] - self.prediction[0]
		RMSE = math.sqrt(sum(err ** 2)/len(err))
		print 'testing set RMSE = %.2f' % (RMSE)

	def main(self):
		self.testing_set_id = self.testing_set['id']
		ols_data = self.testing_set
		self.prediction = self.model.predict(ols_data)
		self.testing_set_RMSE()

DIAG = model_diagnosis('model.pickle', 'test.csv')
DIAG.main()