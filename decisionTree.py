# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 20:06:35 2016

@author: ceciliaLee
"""
from math import log
import operator


## create dataset
def createDataSet():
    # 2 features, 2 classes
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels


## Calculate entropy
def calEntropy(dataSet):
    entry_count=len(dataSet) ## count the number of instances
    print 'entry_count: ',entry_count
    print 'current dataset is: ', dataSet 
    classCountDic={}
    ## create dictionary for all probable classes
    for featureVec in dataSet:
        currentLabel=featureVec[-1] ## the key of class-dictionary is value of the last column
        if currentLabel not in classCountDic.keys(): 
            classCountDic[currentLabel]=0
        classCountDic[currentLabel]+=1
    entropy=0.0
    for key in classCountDic:
        prob=float(classCountDic[key])/entry_count # accuring probability of classes
        entropy-=prob*log(prob,2) # 以2为底
    print "classCountDic, {label:count}:", classCountDic
    print 'the entropy is: ', entropy
    print 
    return entropy
    
    
## Split dataset, select the best feature with  for classifying
def splitDataSet(dataSet,feature,value):
    copyDataSet=[]
    for featureVec in dataSet:
        if featureVec[feature]==value:
            reducedFeatVec=featureVec[:feature]
            reducedFeatVec.extend(featureVec[feature+1:])
            copyDataSet.append(reducedFeatVec)
    return copyDataSet
    
## choosing the best way of splitting dataset
def chooseBestFeatureSplitWay(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calEntropy(dataSet) ## 计算最初的information entropy，用于与split feature dataset之后的information entropy进行compare
    bestInfoGain = 0.0
    bestFeature = -1
    ## 遍历当前特征中的所有唯一属性值，对每个特征划分一次dataset，then 计算dataset的new entropy，and sum unique 特征value得到的entropy
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet] #create a list of all the examples of this feature
        uniqueVals = set(featList)       # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calEntropy(subDataSet)     
        infoGain = baseEntropy - newEntropy     # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       # compare this to the best gain so far
            bestInfoGain = infoGain         # if better than current best, set to best
            bestFeature = i
    print 'the best feature to split dataset is: ', bestFeature,'-th feature'
    return bestFeature                      # returns an integer

## if all features in the dataset have been propcessed，but class labels still are not unique
## then we use 多数表决法 decide the class of leaf node
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): 
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]    


### Constructing the decision tree recursively
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] # the last colunmn is the label of class
    # 如果所有额classes都为相同label，直接返回该类label
    if classList.count(classList[0]) == len(classList): 
        return classList[0]  # stop splitting when all of the classes are equal
        
    # 使用完了所有特征仍不能将dataset划分为仅包含唯一类别的分组，则停止
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureSplitWay(dataSet)
 #   print 'bestFea:', bestFeat
 #   print 'label:',labels
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels) ## recursion
    print
    print 'The constructed decision tree is: ', myTree
    return myTree      
    
            
    