# -*- coding: utf-8 -*-
"""
Created on Thu May 05 15:42:06 2016

@author: selinang
"""

# 20160505  
# Multivariate regression of diamond dataset 
# Model Form:
# math.sqrt(Price) ~ Carat + Carat:Cutting:Color:Clarity
# arguments: <input_file> <train-test_split>
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import math

pth     = "C:\\SNGPrivate\\scripts\\jewellery-pricing\\"
#inf     = "diamonds.csv"
inf     = "diamonds_synthetic.csv"
jewel   = pd.read_csv(pth+inf)

# jewel.columns.values
#   array(['id', 'carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z'], dtype=object)
# jewel.dtypes
#    id           int64
#    carat      float64
#    cut         object
#    color       object
#    clarity     object
#    depth      float64
#    table      float64
#    price        int64
#    x          float64
#    y          float64
#    z          float64
#    dtype: object
results = smf.ols('np.sqrt(price) ~ carat + carat:clarity:color:cut', data=jewel).fit()
#   ValueError: For numerical factors, num_columns must be an int
# sm.show_versions()
# upgraded patsy from 0.4.0 to 0.4.1
print results.summary()
#print jewel['price'][:5]
#print results.predict(jewel.loc[:5,]) ** 2
pred = results.predict(jewel) 
pred[pred<math.sqrt(min(jewel['price']))] = math.sqrt(min(jewel['price']))
pred[pred>math.sqrt(max(jewel['price']))] = math.sqrt(max(jewel['price']))

err = jewel['price'] - (pred ** 2)
print math.sqrt(sum(err ** 2)/len(err))
# RMSE 692.49
# after cap within range of price: RMSE 623.49

results = smf.ols('np.sqrt(price) ~ carat + carat:clarity:color:cut + carat:table:depth', data=jewel).fit()
# RMSE 691.51
# after cap within range of price: RMSE 622.53

results = smf.ols('np.sqrt(price) ~ carat + clarity + carat:clarity:color:cut', data=jewel).fit()
# 665.30
# 608.67

#jewel['carat'] = jewel['carat'] ** (1./3)
#results = smf.ols('np.log(price) ~ carat + carat:clarity:color:cut', data=jewel).fit()
#pred = results.predict(jewel) 
#err = jewel['price'] - np.round(np.exp(pred))

results = smf.ols('np.sqrt(price) ~ carat + carat:clarity:color:cut + shape', data=jewel).fit()
# m$residuals/dts$carat, breaks=c(-25, -15, -10, 10, 15, 90)
# 645.66
# 589.77
# m$residuals, breaks=c(-60, -15, -10, 10, 15, 40)
# 467.34
# 452.51
# m$residuals, breaks=c(-60, -12, -7, 7, 12, 40)
# 395.97
# 379.72
# m$residuals, breaks=c(-60, rmed-2*rsd, rmed-rsd, rmed+rsd, rmed+2*rsd, 40)
# 303.78
# 273.77
# m$residuals, breaks=c(-60, rmed-rsd, rmed-.5*rsd, rmed, rmed+.5*rsd, 40)
# 364.68
# 317.94
# m$residuals, breaks=c(-60, rmed-rsd, rmed-.5*rsd, rmed+.5*rsd, rmed+rsd, 40)
# 340.91
# 293.16


results = smf.ols('np.sqrt(price) ~ carat + carat:clarity:color:cut:shape', data=jewel).fit()
# m$residuals, breaks=c(-60, rmed-2*rsd, rmed-rsd, rmed+rsd, rmed+2*rsd, 40)
# 278.92
# 270.83
# m$residuals, breaks=c(-60, rmed-rsd, rmed-.5*rsd, rmed, rmed+.5*rsd, 40)
# 296.16
# 280.29
# m$residuals, breaks=c(-60, rmed-rsd, rmed-.5*rsd, rmed+.5*rsd, rmed+rsd, 40)
# 280.54
# 263.75