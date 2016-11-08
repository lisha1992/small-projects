# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 18:46:47 2016

@author: ceciliaLee
Steps: 1) transform lists of text into a vector of numbers.
       2) calculate conditional probabilities from these vectors
       3) create a classifier
"""
from numpy import *


## Prepare: making word vectors from text
def load_DataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]## training data
                 
    classVec = [0,1,0,1,0,1]    # labels of training data: 1 is abusive, 0 not
    return postingList,classVec # return a tokenized set of documents and a set of class labels
    
## create a list of all the unique words in all of our documents
def create_vocabList(dataSet):
    vocabSet=set([]) #create empty set
    for doc in dataSet:
        vocabSet=vocabSet | set(doc)  #union of the two sets
    return list(vocabSet)

## Create a feature for each vocabullary list
def setWord2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)  # create a vector of all 0s
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec
    
## Train: calculating probabilities from word vectors(Naive Bayes)
## Pseducode:
# Count the number of documents in each class 
# for every training document
#   for each class:
#       if a token appears in the document ➞ increment the count for that token 
#       increment the count for tokens
#   for each class: 
#       for each token:
#           divide the token count by the total token count to get conditional probabilities 
#   return conditional probabilities for each class

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # like p(w0|1)p(w1|1)p(w2|1). If any of these numbers are 0, 
    # then when we multiply them together we get 0. 
    # To lessen the impact of this, we’ll initialize 
    # all of our occurrence counts to 1, and we’ll initialize 
    # the denominators to 2.
    p0Num = ones(numWords); p1Num = ones(numWords)      # initialize probabilities
    p0Denom = 2.0; p1Denom = 2.0                        # 
    for i in range(numTrainDocs): # for every training document
        if trainCategory[i] == 1:  # for each class
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)         
    p0Vect = log(p0Num/p0Denom) 
## When we go to calculate the product p(w0|ci)p(w1|ci)p(w2|ci)...p(wN|ci) 
 # and many of these numbers are very small, we’ll get underflow, or an 
 # incorrect answer. (Try to multiply many small numbers in Python.
 # Eventually it rounds off to 0.) One solution to this is to take the 
 #  natural logarithm of this product.         
    return p0Vect,p1Vect,pAbusive

## Naive Bayes classifier
def classifierNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # add up the values for all of the words in our vocabulary and 
    # add this to the log probability of the class.
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise multiplication
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
## Tresting         
def testingNB():
    listOPosts,listClasses = load_DataSet()
    myVocabList = create_vocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setWord2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setWord2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifierNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setWord2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifierNB(thisDoc,p0V,p1V,pAb)   
    
## Bag-os-words Model
def bagOfWords2VecMN(vocabList, inputSet): ## Slight change from setWord2Vec() function 
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec






    
listOPosts, listClasses=load_DataSet()

vocabuList=create_vocabList(listOPosts)
print 'vocabuList: ' , vocabuList
for i in range(len(listOPosts)):
    print setWord2Vec(vocabuList,listOPosts[i])
    
trainMat=[]
for postinDoc in listOPosts:#populates the trainMat list with word vectors.
    trainMat.append(setWord2Vec(vocabuList,postinDoc))
# get the proba- bilities of being abusive and the two probability vectors
p0_v, p1_v, p_abu=trainNB0(trainMat,listClasses)
print 'p0_v: ', p0_v
print 'p1_v: ', p1_v
print 'p0_v: ', p_abu
print 'Test the Naive Bayes Classifier:'
testingNB()

    
    