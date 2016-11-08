# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:53:23 2016

@author: ceciliaLee
"""

import matplotlib.pyplot as plt

## define textfield and arrow format
decisionNode=dict(boxstyle='sawtooth',fc='0.8') # boxstyle = "swatooth"意思是注解框的边缘是波浪线型的，fc控制的注解框内的颜色深度  
leafNode=dict(boxstyle='round4',fc='0.8')
arrow_args=dict(arrowstyle='<-') ## 箭头符号


############### using text to annotate nodes ############### 
## plotNode()函数用于绘制箭头和节点，该函数每调用一次，将绘制一个箭头和一个节点
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    ## centerPt: 起点位置（y，x）
    ## parentPt: 箭头位置（y，x）
    ## nodeTxt： 箭头处文本框显示的注解
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',
    xytext=centerPt,textcoords='axes fraction', va='center', ha='center',bbox=nodeType,arrowprops=arrow_args)
    ## xy=parentPt xycoords='axes fraction' 起点位置，  
    ## xytext=centerPt,textcoords='axes fraction' 注解框位置
    ## createPlot.ax1 定义一个绘图区域
def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1=plt.subplot(111,frameon=False)
    plotNode('A decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('A leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
    
    
############### get the number of leaf node and hierachies of tree ############### 
## getLeafsCount(myTree) 遍历整棵树，累计叶子节点的个数
def getLeafsCount(myTree):
    leafCount=0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 如果子节点也是字典类型，则该节点也是判断节点，递归调用 getLeafsCount（）函数
        if type(secondDict[key]).__name__=='dict': # test whether the data type of node is dictionary or not
            leafCount+=getLeafsCount(secondDict[key])
        else:
            leafCount+=1
    return leafCount
        
## getTreeDepth(myTree) 计算遍历过程中的遇到判断节点的个数。遇到 leaf node 就 terminal，return，depth＋1；
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=myTree.keys()[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth
    
## retrieveTree() 输出 pre-saved的 tree information，avoid creating tree from data each time
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]
   
## insert text info between parent and child
## 用于绘制剪头线上的标注，涉及坐标计算，其实就是两个点坐标的中心处添加标注
def insertTextPC(childPt,parentPt,txtInfo):
    xCoor=(parentPt[0] - childPt[0]) / 2.0 + childPt[0]
    yCoor=(parentPt[1] - childPt[1]) / 2.0 + childPt[1]
    createPlot.ax1.text(xCoor, yCoor, txtInfo, va="center", ha="center", rotation=30)
    

## plotTree(tree,parentTree,info) 实现整个树的绘制逻辑和坐标运算，使用的递归，重要的函数
## 画decision逻辑：
## 利用整棵树的叶子节点数作为份数将整个x轴的长度进行平均切分，
#  利用树的深度作为份数将y轴长度作平均切分，并利用plotTree.xOff作为最近绘
#  制的一个叶子节点的x坐标，当再一次绘制叶子节点坐标的时候才会plotTree.xOff
#  才会发生改变;用plotTree.yOff作为当前绘制的深度，plotTree.yOff是在每递
#  归一层就会减一份（上边所说的按份平均切分），其他时候是利用这两个坐标点去计算
#  非叶子节点，这两个参数其实就可以确定一个点坐标，这个坐标确定的时候就是绘制节点的时候
## 
## 3 steps:
#  1) 绘制自身
#  2) 判断子节点－> 若为 non-leafnode -> recursion
#  3) 判断子节点－> 若为 leafnode -> draw the tree

def plotTree(myTree,parentPt,info):
    ## count the number of leafs and tree depth
    leafCount=getLeafsCount(myTree)
    treeDepth=getTreeDepth(myTree)
    childTxt = myTree.keys()[0]  # the text label of this node
    ## 两个全局变量plotTree.xOff和plotTree.yOff用于追踪已绘制的节点位置，并放置下个节点的恰当位置
    ## plotTree.totalW: the width of tree
    ## plotTree.totalD: the depth of tree
    # plotTree.totalW and plotTree.totalD 计算tree nodes的摆放位置，这样可以将tree绘制在水平和垂直方向的中心位置
    # plotTree.totalW 用于计算放置判断节点的位置，主要原则是放在所有叶子节点的中间
    
    childPt=(plotTree.xOff + (1.0 + float(leafCount)) / 2.0/plotTree.totalW, plotTree.yOff)
    insertTextPC(childPt,parentPt,txtInfo)
    
    plotNode(childTxt,childPt,parentPt,decisionNode)
    secondDict = myTree[childTxt]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  ## reduce the y 偏移

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],childPt,str(key))        #recursion
        else:     #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), childPt, leafNode)
            insertTextPC((plotTree.xOff, plotTree.yOff),childPt,str(key))
        plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD  ## 按比例减少 plotTree.yOff
#if you do get a dictonary you know it's a tree, and the first element will be another dict
        
## crete Decision tree  (draw)
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False)    # no ticks
    plotTree.totalW = float(getLeafsCount(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()
    

        
    
    
    
    
    
    
    
    
    