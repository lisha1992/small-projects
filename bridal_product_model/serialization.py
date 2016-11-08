import statsmodels.api as sm
from statsmodels.regression.linear_model import OLSResults
import numpy as np

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)
X = sm.add_constant(X)
y = np.dot(X, beta) + e
model = sm.OLS(y, X)
results = model.fit()
results.save("example.pickle")
new_results = OLSResults.load("example.pickle")
print(new_results.summary())