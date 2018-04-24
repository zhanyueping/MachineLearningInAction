#!/user/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open(r'machinelearninginaction3x-master\Ch05\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxcycle = 500
    weights = np.ones((n,1))
    for k in range(maxcycle):
    #采用批量梯度上升算法
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights

def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stoGradAscent0(dataMatrix,classLabels):
    """随机梯度上升"""
    m,n = np.shape(dataMatrix)
    alpha = 0.1
    weights = np.ones(n)*1.0
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """修正后的随机梯度上升"""
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(np.random.uniform(0, len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def stocGradAscent1_1(dataMatrix, classLabels, numInter=150):
    """作者的代码与他表达的意思不一致，进行修正后的随机梯度上升"""
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numInter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(np.random.uniform(0,len(dataIndex)))
            realChosen = dataIndex[randIndex]
            h = sigmoid(sum(dataMatrix[realChosen]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classfiyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open(r'machinelearninginaction3x-master\Ch05\horseColicTraining.txt')
    frTest = open(r'machinelearninginaction3x-master\Ch05\horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingLabels.append(float(currLine[len(currLine)-1]))
        trainingSet.append(lineArr)
    trainingWeights = stocGradAscent1_1(np.array(trainingSet),trainingLabels,1000)
    errorCount = 0
    numTestVect = 0.0
    for line in frTest.readlines():
        numTestVect += 1
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if (classfiyVector(np.array(lineArr),trainingWeights)
                != currLine[len(currLine)-1]):
            print("%f != %f"
                  %(classfiyVector(np.array(lineArr),trainingWeights),
                    float(currLine[len(currLine)-1])))
            errorCount += 1
    errorRate = float(errorCount)/numTestVect
    print("the error rate of this test is: %f" %errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f"
          %(numTests,errorSum/float(numTests)))











if __name__ == '__main__':
    multiTest()