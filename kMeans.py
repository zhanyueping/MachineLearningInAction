#!/user/bin/env python
# -*- coding:utf-8 -*-

from numpy import *


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA,vecB):
    # 计算两点的欧式距离
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataSet,k):
    """
    构建一个包含k个随机质心的集合
    :param dataSet:
    :param k:
    :return:
    """
    #特征值的总个数
    n = shape(dataSet)[1]
    # 构建簇
    centroids = mat(zeros((k,n)))
    #遍历所有样本,构建簇质心
    for j in range(n):
        minJ = min(dataSet[:,j])
        # 求某个特征值的变动范围
        rangeJ = float(max(dataSet[:,j]) - minJ)
        # 求某个特征值的簇质心---random.rand(k,1)生成了k维1列的随机数
        centroids[:,j] = minJ + rangeJ*random.rand(k,1)
    return centroids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    """
    k-means
    :param dataSet:
    :param k:
    :param distMeas:计算距离
    :param createCent:创建初始质心
    :return:
    """
    m = shape(dataSet)[0]
    #存储每个点的簇分配结果，第一列为簇索引值，第二列为误差，即当前点到簇质心的距离
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        #遍历全部样本
        for i in range(m):
            minDist = inf
            minIndex = -1
            #遍历所有簇，寻找最近质心
            for j in range(k):
                disJI = distMeas(centroids[j,:],dataSet[i,:])
                if disJI < minDist:
                    minDist = disJI
                    minIndex = j
            # 判断样本点的分类是否发送了改变
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        # 更新每个簇中的质心
        for cent in range(k):
            #筛选出簇为cent的全部点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids, clusterAssment

def biKmeans(dataSet,k,distMeas=distEclud):
    """
    二分K-均值算法
    :param dataSet:
    :param k:
    :param distMeas:
    :return:
    """
    #样本总个数
    m = shape(dataSet)[0]
    #存储分类结果，第一列为簇索引值，第二列为误差，即当前点到簇质心的距离
    clusterAssment = mat(zeros((m,2)))
    #初始的大簇质心
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    #保存所有的质心
    centList = [centroid0]
    #计算每个点到大簇的误差
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2
    #循环划分数据，直至簇数目为k
    while (len(centList) < k):
        lowestSSE = inf
        #循环整个簇分类
        for i in range(len(centList)):
            #筛选出簇i中的所有点
            ptsInCurrCluster = \
                dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            #对簇i进行2簇划分
            centroidMat,splitClustAss = \
                kMeans(ptsInCurrCluster,2,distMeas)
            #计算簇i划分后的总误差
            sseSplit = sum(splitClustAss[:,1])
            #计算非簇i中所有点的总误差
            sseNotSplit = \
                sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],:])
            print("sseSplit,and not split:",sseSplit,sseNotSplit)
            if (sseSplit+sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #簇i划分后生成了新的子簇，编号为0和1,因此需要将簇编号重新修改
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        #添加新划分簇的质心
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        #新簇分配结果更新clusterAssment
        clusterAssment[nonzero(clusterAssment[:,0].A == \
                               bestCentToSplit)[0],:] = bestClustAss
    return mat(centList),clusterAssment











def example():
    #第一个例子
    datMat = mat(loadDataSet(r'machinelearninginaction3x-master\Ch10'
                             r'\testSet.txt'))
    myCentroids,clustAssing = kMeans(datMat,4)
    #第二个例子
    datMat2 = mat(loadDataSet(r'machinelearninginaction3x-master\Ch10'
                             r'\testSet2.txt'))
    centList, myNewAssments = biKmeans(datMat2,3)






if __name__ == '__main__':
    example()




