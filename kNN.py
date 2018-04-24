# -*- coding:utf-8 -*-
"""
整个代码中Set 样本集合，前面是特征值，最后一列是label
    即数据是一种由列表元素组成的列表，所有列表元素要有相同的数据长度
    数据的最后一列或者每个实例的最后一个元素为当前实例的分类标签
"""



from numpy import *
import operator
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #生成与dataSetSize同维度的矩阵，并做减法
    diffMat = tile(inX, (dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    #axis=1，注意在第二维度上求和
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #argsort处理后获取的是对值排序后的index，从小到大
    sortedDistIndicies = sqDistances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    #这里根据文件内容进行设置矩阵的大小，文件最后一列是Label
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def file2matrix_pandas(filename):
    #一般文件较小的时候，可以考虑采用pandas，一行代码就能读取文件内容
    returnMat = pd.read_table(filename,header=None)
    returnMat.columns = ['flyingMiles','playingPercent',
                         'icecreamWeight','classLabel']
    classLabelVector = returnMat['classLabel']
    returnMat = returnMat.drop('classLabel',axis=1)
    return returnMat,classLabelVector

def plot_scatter_diagram(datingDataMat,datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2],
               10.0*array(datingLabels),10.0*array(datingLabels))
    plt.show()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(222)
    ax2.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
               10.0 * array(datingLabels), 10.0 * array(datingLabels))
    plt.show()

def autoNorm(dataSet):
    """数据归一化/标准化"""
    #dataSet.min(0)中参数0使得函数能从列中选择最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = \
        file2matrix(r'D:\Desktop\MachineLearningInAction'
                    r'\machinelearninginaction3x-master'
                    r'\Ch02\datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],4)
        print("the calssifierResult is: %d, the real answer is: %d"
              %(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is %f" %(errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all','in small','in large doses']
    percenTats = float(input(
        "percentage of time spent playing video games?"
    ))
    ffmiles = float(input("frequent flier miles earned per years?"))
    icecream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLables = \
        file2matrix(r'D:\Desktop\MachineLearningInAction'
                    r'\machinelearninginaction3x-master'
                    r'\Ch02\datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffmiles,percenTats,icecream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,
                                 datingLables,4)
    print("You will probably like this person: %s"
          %resultList[classifierResult - 1])

def im2vector(filename):
    """将文件中32*32的二进制文件转存为1*1024的向量"""
    returnVector = zeros((1,1024))
    fr = open(filename,encoding='utf-8')
    for i in range(32):
        lineStr = fr.readline().strip()
        for j in range(32):
            returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector

def hadwritingClassTest():
    hwLabels = []

    trainingfileroot = (r'D:\Desktop\MachineLearningInAction'
                          r'\machinelearninginaction3x-master\Ch02'
                          r'\trainingDigits')
    trainingfilelist = os.listdir(trainingfileroot)
    m = len(trainingfilelist)
    trainingData = zeros((m,1024))
    for i in range(m):
        filename = trainingfilelist[i]
        trainingData[i, :] = im2vector(trainingfileroot+"\\"+filename)
        label = int(filename.split(r'\\')[-1][0])
        hwLabels.append(label)

    testfileroot= (r'D:\Desktop\MachineLearningInAction'
                    r'\machinelearninginaction3x-master\Ch02'
                    r'\testDigits')
    testfilelist = os.listdir(testfileroot)
    n = len(testfilelist)
    errorcount = 0.0
    for i in range(n):
        filename = testfilelist[i]
        test = im2vector(testfileroot+"\\"+filename)
        testlabel = int(filename.split(r'\\')[-1][0])
        classifierlabel = classify0(test,trainingData,hwLabels,3)
        if (classifierlabel != testlabel): errorcount += 1.0
    print("the total number of errors are: %d" %errorcount)
    print("the total error rate is: %f" %(errorcount/float(n)))















if __name__ == '__main__':

    hadwritingClassTest()