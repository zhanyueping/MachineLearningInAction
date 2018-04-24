#!/user/bin/env python
# -*- coding:utf-8 -*-

from numpy import *
import matplotlib
import matplotlib.pyplot as plt


def loadDataSet(fileName,delim='\t'):
    """通过文件获取数据

    :param fileName: 文件路径
    :param delim: 文件分隔符
    :return:
    """
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat = 9999999):
    """
    PCA降维
    :param dataMat:
    :param topNfeat:
    :return:
    """
    #求平均值
    meanVals = mean(dataMat,axis=0)
    #标准化处理
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵
    covMat = cov(meanRemoved,rowvar=0)
    #计算协方差矩阵的特征值和特征向量
    eigVals,eigVects = linalg.eig(mat(covMat))
    #对特征值排序,从小到大,获取对应的index
    eigValInd = argsort(eigVals)
    #选择最大的前topNfeat个特征值
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    #降维
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T)+meanVals
    return lowDDataMat,reconMat


def replaceNanWithMean():
    datMat = loadDataSet('./machinelearninginaction3x-master'
                          '/Ch13/secom.data')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        #计算非空特征值的均值
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        #用均值代替空值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

def example():
    #第一个例子
    dataMat = loadDataSet('./machinelearninginaction3x-master'
                          '/Ch13/testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0],
               reconMat[:, 1].flatten().A[0],
               marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0],
               reconMat[:, 1].flatten().A[0],
               marker='o', s=50, c='red')
    plt.show()

    #第2个例子
    dataMat = replaceNanWithMean()
    lowDMat, reconMat = pca(dataMat, 6)


if __name__ == '__main__':
    example()