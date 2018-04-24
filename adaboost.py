#!/user/bin/env python
# -*- coding:utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

def loadSimpleData():
    dataMat = matrix([[1.0,2.1],[2.0,1.1],
                      [1.3,1.0],[1.0,1.0],
                      [2.0,1.0]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threShIneq):
    """
    通过阈值比较进行数据分类
    :param dataMatrix:
    :param dimen:
    :param threshVal:
    :param threShIneq:
    :return:
    """
    retArray = ones((shape(dataMatrix)[0],1))
    if threShIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    """
    单层决策树
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numStemps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    #对数据集的全部特征值进行遍历
    for i in range(n):
        #取某个数值型特征值的最大值
        rangeMin = dataMatrix[:,i].min()
        # 取某个数值型特征值的最小值
        rangeMax = dataMatrix[:,i].max()
        #确定步长
        stepSize = (rangeMax - rangeMin)/numStemps
        #遍历该数值型特征值的全部值
        for j in range(-1,int(numStemps)+1):
            # 在大于和小于阈值之间切换，寻找最优分类
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0

                weightedError = D.T*errArr
                print("split: dim %d, thresh %.2f, thresh inequal: %s,"
                      "the weighted error is %.3f" %(i,threshVal,inequal,weightedError))

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    """
    基于单层决策树的AdaBoost训练过程
    :param dataArr:
    :param classLabels:
    :param numIt: 训练次数
    :return:
    """
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    # 记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print("D:",D.T)
        #计算alpha，其中max()作用是防止error为0
        alpha =  float(0.5*log((1.0-error)/max(error,1e-16)))
        #存储alpha
        bestStump['alpha'] = alpha
        #存储最优选择
        weakClassArr.append(bestStump)
        print("classEst:",classEst.T)
        #这里就明白为classLabels中2个类别设置为1和-1
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print("aggclassEst: ",aggClassEst)
        aggErrors = multiply(sign(aggClassEst) !=
                             mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(datToClass,classfierArr):
    """
    分类
    :param datToClass:
    :param classfierArr:
    :return:
    """
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classfierArr)):
        classEst = stumpClassify(dataMatrix,classfierArr[i]['dim'],
                                 classfierArr[i]['thresh'],
                                 classfierArr[i]['ineq'])
        aggClassEst += classfierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)

def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))  # get number of fields
    dataMat = []
    labelMat = []
    fr = open(filename)

    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    fr.close()
    return dataMat,labelMat

def plotROC(predStrenths,classLabels):
    """
    画ROC曲线图
    :param predStrenths:分类器的预测强度
    :param classLabels: 真实分类情况
    :return:
    """
    #一开始全部预测为真，1为真，0为假
    cur = (1.0,1.0)
    ySum = 0.0
    #计算全部样本中正例的数目
    numPosClas = sum(array(classLabels) == 1.0)
    #步长
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    #从小到大排序，得到索引值
    sortedIndicies = predStrenths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #遍历所有值，预测为假的样本数目不断增多
    for index in sortedIndicies.tolist()[0]:
        #当标签为1.0，y轴下降一个步长，不断降低真阳率
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
        #在X轴倒退一个步长，不断降低假阴率
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    print("the Area Under the Curve is: ", ySum * xStep)
    plt.show()












if __name__ == '__main__':

    datMat,classLabels = loadDataSet(r'machinelearninginaction3x-master'
                                     r'\Ch07\horseColicTraining2.txt')
    print(len(classLabels))

    classifierArr,aggClassEst = adaBoostTrainDS(datMat,classLabels,10)
    plotROC(aggClassEst.T,classLabels)
    # testArr,testLabelArr = loadDataSet(r'machinelearninginaction3x-master'
    #                                  r'\Ch07\horseColicTest2.txt')
    # print(len(testLabelArr))
    # prediction10 = adaClassify(testArr,classifierArr)
    # errArr = mat(ones((67,1)))
    # errRate = errArr[prediction10 != mat(testLabelArr).T].sum()
    #
    # print(errRate)
    # print(weakClassArr)



