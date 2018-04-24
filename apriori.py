#!/user/bin/env python
# -*- coding:utf-8 -*-

from numpy import *

def loadDataSet():
    #创建初始数据，不同数字代表不同物品，一个子集合代表一个项值
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    #创建存储所有不重复只有一个元素的项值
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                #添加项值列表
                C1.append([item])
    C1.sort()
    #映射到frozenset()，最后返回该列表
    return list(map(frozenset,C1))

def scanD(D,Ck,minSupport):
    """
    筛选符合要求的项值
    :param D:数据集
    :param Ck:候选列表
    :param minSupport:最小支持度
    :return:
    """
    #存储项集（key）和它在整个数据集中出现的次数(value)
    ssCnt = {}
    #遍历数据集
    for tid in D:
        #遍历候选列表
        for can in Ck:
            #判断是否为数据集中的子集
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    #数据集中总记录数目，即总项集数目
    numItems = float(len(D))
    #存储满足要求的项集
    retList = []
    #最频繁项集的支持度
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData

def aprioriGen(Lk,k):
    """
    创建候选项集
    :param Lk:频繁项集
    :param k:项集元素个数k
    :return:
    """
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            # 取两个集合前面k-2个元素，进行判断
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            #两个集合前面k-2个元素都相等，则合并
            if L1 == L2:
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataSet,minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    #循环递增k，直到最后Lk为空，退出循环
    while (len(L[k-2]) > 0 ):
        Ck = aprioriGen(L[k-2],k)
        Lk,supK = scanD(D,Ck,minSupport)
        #把字典supK的键/值对更新到supportData里
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData

def calcConf(freqSet,H,supportData,brl,minConf=0.7):
    """
    遍历H中所有项集，并计算它们的可信度，返回一个满足最小可信度的规则列表
    :param freqSet:频繁项集
    :param H:规则右部的元素列表H
    :param supportData:
    :param br1:
    :param minConf:
    :return:
    """
    #保存符合要求的规则
    prunedH = []
    #遍历H，计算规则的可信度
    for conseq in H:
        #计算规则的可信度：freqSet-conseq ——> conseq
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet,H,supportData,brl,minConf=0.7):
    """
    采用递归，生成候选规则集合，并进行筛选
    :param freqSet:频繁项集
    :param H:规则右部元素列表
    :param supportData:
    :param brl:
    :param minConf:
    :return:
    """
    #计算H中频繁项集的个数
    m = len(H[0])
    #若freqSet中频繁项集数目大于H的
    if (len(freqSet) > (m+1)):
        #生成H中元素的无重复组合
        Hmpl = aprioriGen(H,m+1)
        #计算Hmpl中符合要求的规则
        Hmpl = calcConf(freqSet,Hmpl,supportData,brl,minConf=0.7)
        #若有符合要求的规则，则需要继续迭代，看能否继续组合这些规则
        if(len(Hmpl) > 1):
            rulesFromConseq(freqSet,Hmpl,supportData,brl,minConf)

def generateRules(L,supportData,minConf=0.7):
    """
    生成规则
    :param L:频繁项集列表
    :param supportData:包含频繁项集支持度数据的字典
    :param minConf:最小可信度
    :return:一个包含可信度的规则列表
    """
    bigRuleList = []
    #遍历L中每个频繁项集，i从1开始，即从包含2个的项集开始
    for i in range(1,len(L)):
        #遍历每个频繁项集中的元素列表
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            #从包含至少2个的项集开始构建规则
            if ( i > 1):
                #若包含2个以上元素的项集，则可以生成更多关联规则
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                #若只包含2个项集，直接计算可信度
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList
























def example():
    #第一个例子
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    D = list(map(set,dataSet))
    L1,suppData0 = scanD(D,C1,0.5)
    print(L1)
    #第二个例子
    L,suppData = apriori(D)
    print(L)
    #第三个例子
    rules = generateRules(L,suppData,minConf=0.7)
    print("rules")


if __name__ == '__main__':
    example()