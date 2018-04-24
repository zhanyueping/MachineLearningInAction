#!/user/bin/env python
# -*- coding:utf-8 -*-

from numpy import *
from tkinter import *




def loadDataSet(fileName):
    #存储文件里的数据
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    """
    二元法切分数据集
    :param dataSet: 数据集
    :param feature: 待切分的特征
    :param value: 阈值
    :return:
    """
    #大于阈值，进入左子树
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    #小于阈值，进入右子树
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    #生成叶节点---目标变量的均值
    return mean(dataSet[:,-1])

def regErr(dataSet):
    #总方差计算
    return var(dataSet[:,-1])*shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    """
    找到数据集切分的最佳位置
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops:
    :return:
    """
    #容许的误差下降值
    tolS = ops[0]
    #切分的最少样本数
    tolN = ops[1]
    #判断样本数目
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    #循环数据中全部特征
    for featIndex in range(n-1):
        # 循环特征的全部可能取值
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0]<tolN) or (shape(mat1)[0]<tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #若下降误差比阈值小，即误差变化不大，则退出
    if (S - bestS) < tolS:
        return None,leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    #若切分后样本数目小于阈值
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0]<tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue

def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    """
    建树
    :param dataSet:数据集
    :param leafType:叶节点函数
    :param errType:误差计算函数
    :param ops:包含树构建所需的其他参数元组
    :return:
    """
    #寻找最佳切分
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

def isTree(obj):
    #判断是否为树，即字典
    return (type(obj).__name__=='dict')

def getMean(tree):
    # 递归，找到2个叶节点并计算它们的均值
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right']/2.0)

def prune(tree,testData):
    """
    剪枝
    :param tree: 待剪枝的树
    :param testData: 测试数据
    :return:
    """
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right'])) or (isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + \
                        sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

def linearSolve(dataSet):
    #生成数据集的目标变量Y和自变量X
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,n)))
    # 将之前线性回归中的b作为一个权重，所以X第一列为1
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    #讨论矩阵是否可逆
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y

def modelLeaf(dataSet):
    #生产叶节点模型
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    #计算误差
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y-yHat,2))

def regTreeEval(model,inDat):
    # 回归树的叶节点预测
    return float(model)

def modelTreeEval(model,inDat):
    # 模型树的叶节点预测
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

def treeForeCast(tree,inData,modelEval=regTreeEval):
    """
    递归预测
    :param tree:
    :param inData:
    :param modelEval:
    :return:
    """
    if not isTree(tree):
        return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

def createForeCast(tree,testData,modelEval=regTreeEval):
    """
    预测检测组数据
    :param tree:
    :param testData:
    :param modelEval:
    :return:
    """
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    #遍历全部样本
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

def example_ex00():
    #第一个例子
    # myDat = loadDataSet(r'machinelearninginaction3x-master\Ch09'
    #                     r'\ex00.txt')
    # myDat = mat(myDat)
    # retTree = createTree(myDat)
    # print(retTree)
    # #第二个例子
    # myDat2 = loadDataSet(r'machinelearninginaction3x-master\Ch09'
    #                     r'\ex0.txt')
    # myDat2 = mat(myDat2)
    # retTree = createTree(myDat2)
    # print(retTree)
    #第三个例子
    # myTree = createTree(myDat2,ops=(0,1))
    # myDatTest = loadDataSet(r'machinelearninginaction3x-master\Ch09'
    #                     r'\ex2test.txt')
    # myMat2Test = mat(myDatTest)
    # prune(myTree,myMat2Test)
    #第四个例子
    myDat3 = loadDataSet(r'machinelearninginaction3x-master\Ch09'
                         r'\exp2.txt')
    myDat3 = mat(myDat3)
    retTree = createTree(myDat3,modelLeaf,modelErr,(1,10))
    print(retTree)

def example_2():
    # 第5个例子
    trainMat = mat(loadDataSet(r'machinelearninginaction3x-master\Ch09'
                               r'\bikeSpeedVsIq_train.txt'))
    testMat = mat(loadDataSet(r'machinelearninginaction3x-master\Ch09'
                               r'\bikeSpeedVsIq_test.txt'))
    myTree = createTree(trainMat,ops=(1,20))
    yHat = createForeCast(myTree,testMat[:,0])
    corrcoefValue = corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    print(corrcoefValue)
    myTree = createTree(trainMat,modelLeaf,modelErr, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0],modelTreeEval)
    corrcoefValue = corrcoef(yHat, testMat[:, 1], rowvar=0)[0,1]
    print(corrcoefValue)
    ws,X = linearSolve(trainMat)[:2]
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i,0]*ws[1,0] + ws[0,0]
    corrcoef(yHat,testMat[:,1],rowvar=0)[0,1]
    print(corrcoefValue)





if __name__ == '__main__':
    example_2()














