#!/user/bin/env python
# -*- coding:utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) -1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegress(xArr,yArr):
    """
    计算回归系数
    :param xArr:
    :param yArr:
    :return:
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    #判断矩阵是否可逆，通过判断矩阵的行列式是否为零，若为0，则不可逆
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
    ws = xTx.I * (xMat.T * yMat)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    """
    局部加权线性回归
    :param testPoint: 某个测试样本
    :param xArr: 全部样本
    :param yArr:
    :param k:
    :return:
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    #创建对角矩阵
    weights = mat(eye((m)))
    #遍历全部样本
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T*(weights*xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    """
    在全部样本中计算权重,并且返回y值
    :param testArr:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr):
    #误差平方和
    return ((yArr-yHatArr)**2).sum()

def example_abalone():
    #书中的真实案例-鲍鱼
    abX, abY = loadDataSet(r'machinelearninginaction3x-master\Ch08'
                           r'\abalone.txt')
    yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    #计算误差平方和
    print(rssError(abY[0:99],yHat01))
    print(rssError(abY[0:99], yHat1))
    print(rssError(abY[0:99], yHat10))
    #计算相关系数
    print(corrcoef(yHat01,abY[0:99]))
    print(corrcoef(yHat1, abY[0:99]))
    print(corrcoef(yHat10, abY[0:99]))
    #检查测试集
    yHat01 = lwlrTest(abX[100:199], abX[100:199], abY[100:199], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[100:199], abY[100:199], 1)
    yHat10 = lwlrTest(abX[100:199], abX[100:199], abY[100:199], 10)
    print(rssError(abY[100:199], yHat01))
    print(rssError(abY[100:199], yHat1))
    print(rssError(abY[100:199], yHat10))

def ridgeRegres(xMat,yMat,lam=0.2):
    """
    岭回归
    :param xMat:
    :param yMat:
    :param lam:
    :return:
    """
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr,yArr):
    #岭回归
    xMat = mat(xArr)
    yMat = mat(yArr).T
    #数据标准化
    yMeans = mean(yMat,0)
    yMat = yMat - yMeans
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat-xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat

def example_abalone2():
    # 书中的真实案例-鲍鱼,采用岭回归的方法
    abX, abY = loadDataSet(r'machinelearninginaction3x-master\Ch08'
                           r'\abalone.txt')
    ridgeWeights = ridgeTest(abX,abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

def regularize(xMat):
    #数据标准化
    inMat = xMat.copy()
    inMeans = mean(inMat,0)
    inVar = var(inMat,0)
    inMat = (inMat - inMeans)/inVar
    return inMat



def stageWise(xArr,yArr,eps=0,numIt=100):
    """
    前向逐步回归
    :param xArr:输入数据
    :param yArr: 预测变量
    :param eps: 每次迭代需要调整的步长
    :param numIt: 迭代次数
    :return:
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    #循环迭代次数
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        #遍历所有特征值
        for j in range(n):
            #增加或减少步长
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

def crossValidation(xArr,yArr,numVal=10):
    """
    交叉验证测试岭回归
    :param xArr:
    :param yArr:
    :param numVal:
    :return:
    """
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal,30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        #混淆索引
        random.shuffle(indexList)
        #创建测试集和训练集
        for j in range(m):
            if j < m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        #创建30个不同的lamda值，从而创建30个不同的回归系数
        wMat = ridgeTest(trainX,trainY)
        #在测试集上用30组回归系数测试回归效果
        for k in range(30):
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain
            yEst = matTestX*mat(wMat[k,:]).T+mean(trainY)
            errorMat[i,k] = rssError(yEst.T.A,array(testY))
    meanErrors = mean(errorMat,0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat,0)
    varX = var(xMat,0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat))










if __name__ == '__main__':
    abX, abY = loadDataSet(r'machinelearninginaction3x-master\Ch08'
                           r'\abalone.txt')
    stageWise(abX, abY,0.001,5000)








