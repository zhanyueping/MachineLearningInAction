#!/user/bin/env python
# -*- coding:utf-8 -*-

from numpy import *

def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def ecludSim(inA,inB):
    """
    采用欧拉距离公式计算A,B两个列向量之间的相似度
    :param inA:
    :param inB:
    :return:
    """
    return 1.0/(1.0+linalg.norm(inA-inB))

def pearsSim(inA,inB):
    """
    采用Pearson相关系数计算A,B两个列向量之间的相似度
    :param inA:
    :param inB:
    :return:
    """
    if (len(inA)<3):
        return 1.0
    return 0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    """
    采用余弦相似度来计算A,B两个列向量之间的相似度
    :param inA:
    :param inB:
    :return:
    """
    num = float(inA.T*inB)
    denom = linalg.norm(inA)*linalg.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataMat,user,simMeas,item):
    """
    基于物品相似度的推荐引擎
    :param dataMat:数据矩阵，行对应用户，列对应物品
    :param user:用户编号
    :param simMeas:相似度计算方法
    :param item:物品编号
    :return:
    """
    #物品的数目
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    #遍历物品，将item和其他物品进行比较
    for j in range(n):
        #用户的评分
        userRating = dataMat[user,j]
        if userRating == 0:
            continue
        #寻找对两个物品都评级的用户
        overLap = nonzero(logical_and(dataMat[:,item].A>0,
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            #计算相似度
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        print("the %d and %d similarity is: %f" %(item,j,similarity))
        #累加相似度
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        #进行归一化处理
        return ratSimTotal/simTotal

def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=standEst):
    """
    推荐引擎
    :param dataMat:数据矩阵，行对应用户，列对应物品
    :param user:用户编号
    :param N:
    :param simMeas:相似度计算方法
    :param estMethod:估计方法
    :return:
    """
    #寻找未评级的物品
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estimatedScore))
    #返回前N个未评级的物品
    return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]

def svdEst(dataMat,user,simMeas,item):
    """
    对给定用户和给定物品构建一个评分估值
    :param dataMat:
    :param user:
    :param simMeas:
    :param item:
    :return:
    """
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,VT = linalg.svd(dataMat)
    #构建对角矩阵
    Sig4 = mat(eye(4)*Sigma[:4])
    #构建转换后的数据
    xformedItems = dataMat.T*U[:,:4]*Sig4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item,:].T,
                             xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' %(item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def printMat(inMat,thresh=0.8):
    """

    :param inMat:
    :param thresh:
    :return:
    """
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1, end=''),
            else:
                print(0, end=''),
            print('')

def imgCompress(numSV=3,thresh=0.8):
    """

    :param numSV:
    :param thresh:
    :return:
    """
    my1 = []
    fr=open(r'.\machinelearninginaction3x-master\Ch14\0_5.txt').readlines()
    for line in fr:
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        my1.append(newRow)
    myMat = mat(my1)
    print("***original matrix***")
    printMat(myMat,thresh)
    U,Sigma,VT = linalg.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):  # construct diagonal matrix from vector
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)






def example():
    #第一个例子
    # data = loadExData()
    # U,Sigma,VT = linalg.svd(data)
    # print(U,Sigma,VT)
    # #重新构建矩阵，因为Sigma前3个值比其他值都大
    # Sig3 = mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[2]]])
    # #重构原始矩阵的近似矩阵，U只取前3列，VT只取前3行
    # data_svd = U[:,:3]*Sig3*VT[:3,:]
    #第2个例子
    myMat = mat(loadExData())
    myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
    myMat[3,3]=2
    recommendArray = recommend(myMat,2)
    print(recommendArray)
    recommendArray = recommend(myMat, 2,simMeas=ecludSim)
    print(recommendArray)
    recommendArray = recommend(myMat, 2, simMeas=pearsSim)
    print(recommendArray)
    #第3个例子
    U,Simga,VT = linalg.svd(mat(loadExData2()))
    Sig2 = Simga**2
    Sigpercent90 = sum(Sig2)*0.9
    for num in range(1,len(Simga),1):
        if sum(Sig2[:num]) > Sigpercent90:
            print(num)
            break
    #第4个例子
    recommend(myMat,1,estMethod=svdEst)
    #第5个例子
    imgCompress(2)
if __name__ == '__main__':
    example()


