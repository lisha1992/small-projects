# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 19:29:55 2016

@author: ceciliaLee
"""
########################  Load data  ######################## 
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
 

df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
 
# Scan dataset
print df.head()
#    admit  gre   gpa  rank
# 0      0  380  3.61     3
# 1      1  660  3.67     3
# 2      1  800  4.00     1
# 3      1  640  3.19     4
# 4      0  520  2.93     4
# [:,:3] are predictor variables
# [:,4] admit is binary target variable，it represents whether the syudent is enrolled or not
 
# 重命名'rank'列，因为dataframe中有个方法名也为'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]
print df.columns
# array([admit, gre, gpa, prestige], dtype=object)



#################### 统计摘要(Summary Statistics) 以及查看数据 ##############
# summarize the data


print df.describe()
#             admit         gre         gpa   prestige
# count  400.000000  400.000000  400.000000  400.00000
# mean     0.317500  587.700000    3.389900    2.48500
# std      0.466087  115.516536    0.380567    0.94446
# min      0.000000  220.000000    2.260000    1.00000
# 25%      0.000000  520.000000    3.130000    2.00000
# 50%      0.000000  580.000000    3.395000    2.00000
# 75%      1.000000  660.000000    3.670000    3.00000
# max      1.000000  800.000000    4.000000    4.00000
 
# 查看每一列的标准差
print df.std()
# admit      0.466087
# gre      115.516536
# gpa        0.380567
# prestige   0.944460
 
# 频率表，表示prestige与admin的值相应的数量关系
# crosstab 可方便的实现多维频率表(frequency tables)
print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])
# prestige   1   2   3   4
# admit                   
# 0         28  97  93  55
# 1         33  54  28  12
 
# plot all of the columns
df.hist()
pl.show()

################## 虚拟变量(dummy variables) #################
# 虚拟变量，也叫哑变量，可用来表示分类变量、非数量因素可能产生的影响。
# 为了反映属性因素和提高模型的精度，必须将属性因素“量化”。通过构造0-1型的人工变量来量化属性因素。

# 将prestige设为虚拟变量
# get_dummies为每个指定的列创建了新的带二分类预测变量的DataFrame
dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')#产生四列的dataframe，每一列表示四个级别中的一个
print dummy_ranks.head()
#    prestige_1  prestige_2  prestige_3  prestige_4
# 0           0           0           1           0
# 1           0           0           1           0
# 2           1           0           0           0
# 3           0           0           0           1
# 4           0           0           0           1
 
# 为逻辑回归创建所需的data frame
# 除admit、gre、gpa外，加入了上面常见的虚拟变量（注意，引入的虚拟变量列数应为虚拟变量总列数减1，减去的1列作为基准）
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
print data.head()
#    admit  gre   gpa  prestige_2  prestige_3  prestige_4
# 0      0  380  3.61           0           1           0
# 1      1  660  3.67           0           1           0
# 2      1  800  4.00           0           0           0
# 3      1  640  3.19           0           0           1
# 4      0  520  2.93           0           0           1
 
# 需要自行添加逻辑回归所需的intercept变量
data['intercept'] = 1.0

################## 执行逻辑回归 ###################

# 指定作为训练变量的列，不含目标列`admit`
train_cols = data.columns[1:] ## training 

# Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)
logit = sm.Logit(data['admit'], data[train_cols])

# 拟合模型
result = logit.fit()


###################### 使用训练模型预测数据 ####################
# 构建预测集
# 与训练集相似，一般也是通过 pd.read_csv() 读入
# 在这边为方便，我们将训练集拷贝一份作为预测集（不包括 admin 列）
import copy
combos = copy.deepcopy(data)
 
# 数据中的列要跟预测时用到的列一致
predict_cols = combos.columns[1:]
 
# 预测集也要添加intercept变量
combos['intercept'] = 1.0
 
# 进行预测，并将预测评分存入 predict 列中
combos['predict'] = result.predict(combos[predict_cols])
 
# 预测完成后，predict 的值是介于 [0, 1] 间的概率值
# 我们可以根据需要，提取预测结果
# 例如，假定 predict > 0.5，则表示会被录取
# 在这边我们检验一下上述选取结果的精确度
total = 0
hit = 0
for value in combos.values:
  # 预测分数 predict, 是数据中的最后一列
  predict = value[-1]
  # 实际录取结果
  admit = int(value[0])
 
  # 假定预测概率大于0.5则表示预测被录取
  if predict > 0.5:
    total += 1
    # 表示预测命中
    if admit == 1:
      hit += 1
 
# 输出结果
print 'Total: %d, Hit: %d, Precision: %.2f' % (total, hit, 100.0*hit/total)
# Total: 49, Hit: 30, Precision: 61.22


#################### 结果解释 (result summary) ###################
# 查看数据的要点
print result.summary()

# 查看每个系数的置信区间
print result.conf_int()
#                    0         1
# gre         0.000120  0.004409
# gpa         0.153684  1.454391
# prestige_2 -1.295751 -0.055135
# prestige_3 -2.016992 -0.663416
# prestige_4 -2.370399 -0.732529
# intercept  -6.224242 -1.755716

#################### 相对危险度（odds ratio）  ################
# 输出 odds ratio
print np.exp(result.params)
# gre           1.002267
# gpa           2.234545
# prestige_2    0.508931
# prestige_3    0.261792
# prestige_4    0.211938
# intercept     0.018500

# 也可以使用置信区间来计算系数的影响，来更好地估计一个变量影响录取率的不确定性。
# odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print np.exp(conf)
#                   2.5%     97.5%        OR
# gre           1.000120  1.004418  1.002267
# gpa           1.166122  4.281877  2.234545
# prestige_2    0.273692  0.946358  0.508931
# prestige_3    0.133055  0.515089  0.261792
# prestige_4    0.093443  0.480692  0.211938
# intercept     0.001981  0.172783  0.018500

################# deeper mining ##################
# 为了评估我们分类器的效果，我们将使用每个输入值的逻辑组合（logical combination）
# 来重新创建数据集，如此可以得知在不同的变量下预测录取可能性的增加、减少。
# 首先我们使用名为 cartesian 的辅助函数来生成组合值

# 根据最大、最小值生成 GRE、GPA 均匀分布的10个值，而不是生成所有可能的值
gres = np.linspace(data['gre'].min(), data['gre'].max(), 10)
print gres
# array([ 220.        ,  284.44444444,  348.88888889,  413.33333333,
#         477.77777778,  542.22222222,  606.66666667,  671.11111111,
#         735.55555556,  800.        ])
gpas = np.linspace(data['gpa'].min(), data['gpa'].max(), 10)
print gpas
# array([ 2.26      ,  2.45333333,  2.64666667,  2.84      ,  3.03333333,
#         3.22666667,  3.42      ,  3.61333333,  3.80666667,  4.        ])
 
 
# 枚举所有的可能性
combos = pd.DataFrame(cartesian([gres, gpas, [1, 2, 3, 4], [1.]]))
# 重新创建哑变量
combos.columns = ['gre', 'gpa', 'prestige', 'intercept']
dummy_ranks = pd.get_dummies(combos['prestige'], prefix='prestige')
dummy_ranks.columns = ['prestige_1', 'prestige_2', 'prestige_3', 'prestige_4']
 
# 只保留用于预测的列
cols_to_keep = ['gre', 'gpa', 'prestige', 'intercept']
combos = combos[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
 
# 使用枚举的数据集来做预测
combos['admit_pred'] = result.predict(combos[train_cols])
 
print combos.head()
#    gre       gpa  prestige  intercept  prestige_2  prestige_3  prestige_4  admit_pred
# 0  220  2.260000         1          1           0           0           0    0.157801
# 1  220  2.260000         2          1           1           0           0    0.087056
# 2  220  2.260000         3          1           0           1           0    0.046758
# 3  220  2.260000         4          1           0           0           1    0.038194
# 4  220  2.453333         1          1           0           0           0    0.179574

## 比较给定的变量与不同的声望等级、组合的平均可能性
def isolate_and_plot(variable):
    # isolate gre and class rank
    grouped = pd.pivot_table(combos, values=['admit_pred'], index=[variable, 'prestige'],
                            aggfunc=np.mean)
 
    # in case you're curious as to what this looks like
    # print grouped.head()
    #                      admit_pred
    # gre        prestige            
    # 220.000000 1           0.282462
    #            2           0.169987
    #            3           0.096544
    #            4           0.079859
    # 284.444444 1           0.311718
 
    # make a plot
    colors = 'rbgyrbgy'
    for col in combos.prestige.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        pl.plot(plt_data.index.get_level_values(0), plt_data['admit_pred'],
                color=colors[int(col)])
 
    pl.xlabel(variable)
    pl.ylabel("P(admit=1)")
    pl.legend(['1', '2', '3', '4'], loc='upper left', title='Prestige')
    pl.title("Prob(admit=1) isolating " + variable + " and presitge")
    pl.show()
 
isolate_and_plot('gre')
isolate_and_plot('gpa')
