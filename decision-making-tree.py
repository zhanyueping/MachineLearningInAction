#! -*- coding: utf-8 -*-

"""
整个代码中Set 样本集合，前面是特征值，最后一列是label
    即数据是一种由列表元素组成的列表，所有列表元素要有相同的数据长度
    数据的最后一列或者每个实例的最后一个元素为当前实例的分类标签
"""
import math
import operator
import matplotlib.pyplot as plt
import pickle

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def calcShannonEnt(dataSet):
    #根据公式计算某个数据集的熵，dataSet最后一列为label
    numEntries = len(dataSet)   #dataSet的样本总数目
    labelCount = {}     #key：类别，value：属于该类别的样本数目
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCount:
        # 用频率代替概率
        prob = float(labelCount[key])/numEntries
        shannonEnt -= prob * math.log(prob,2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    #划分数据集,3个参数分别是：待划分的数据集，划分数据集的特征，需要返回的特征值
    #axis是和value对应，即某个样本第axis列特征的某个值为value
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def createDataSet():
    #临时创建的数据
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
        #计算各种划分方式的信息熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            #计算在某个特征值下的信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
        #计算最好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    #多数表决法
    classCout = {}
    for vote in classList:
        if vote not in classCout.keys(): classCout[vote] = 0
        classCout[vote] += 1
    sortedClassCount = sorted(classCout.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return  sortedClassCount[0][0]

def createTree(dataSet,labels):
    """

    :param dataSet: 数据集
    :param labels: 标签列表
    :return: subTree
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        #类别完全相同
        return classList[0]
    if len(dataSet[0]) == 1:
        #遍历完所有特征
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet
                                                  (dataSet,bestFeat,value),subLabels)
    return myTree

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode("决策节点",(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode("叶节点",(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree)[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else: numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree)[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree)[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

def classify(inputTree,featLabels,testVec):
    """

    :param inputTree:
    :param featLabels:
    :param testVect:
    :return:
    """
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat,dict):
        classLabel = classify(valueOfFeat,featLabels,testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    fw = open(filename,'wb')
    pickle.dump(inputTree,filename)
    fw.close()

def grabTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)





if __name__ == '__main__':

    fr = open(r'D:\Desktop\MachineLearningInAction'
              r'\machinelearninginaction3x-master\Ch03'
              r'\lenses.txt')
    lenses = []
    frlines = fr.readlines()
    for inst in frlines:
        lenses.append(inst.strip().split('\t'))
    lensesLables = ['age','prescript','astigmatic','tearRate']
    lensesTree = createTree(lenses,lensesLables)
    createPlot(lensesTree)