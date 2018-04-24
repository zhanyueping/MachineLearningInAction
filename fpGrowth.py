#!/user/bin/env python
# -*- coding:utf-8 -*-

from numpy import *



class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self,numOccur):
        #增加给定量
        self.count += numOccur

    def disp(self,ind=1):
        #将树以文本的形式展示
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

def createTree(dataSet,minSup=1):
    #构建FP树

    #头指针，存储FP树中单个元素和该元素的出现次数
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            #注意，这里dataSet[trans]=1,所有的trans对应为1。计算每个元素项对应出现次数。
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]
    #字典的元素个数不能一边遍历一边删除
    headerTableCopy = headerTable.copy()
    #遍历全部的单个元素
    for k in headerTableCopy.keys():
        # 删除支持度不满足要求的元素
        if headerTable[k] < minSup:
            del(headerTable[k])
    #存储符合要求的元素项
    freqItemSet = set(headerTable.keys())
    print('freqItemSet: ',freqItemSet)
    #没有符合要求的单个元素项，则退出
    if len(freqItemSet) == 0:
        return None,None
    #对头指针，进行values重置，values为一个list,第一个存储出现元素项次数，第二个存储链接节点
    for k in headerTable:
        headerTable[k] = [headerTable[k],None]
    print('headerTable: ', headerTable)
    #开始建树
    retTree = treeNode('Null Set',1,None)
    #循环遍历样本，只考虑频繁项
    for tranSet,count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                #过滤，删除不符合要求的元素项
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            #根据元素项出现次数排序，从大到小
            orderedItems = [v[0] for v in sorted(localD.items(),
                                                 key=lambda p:p[1],
                                                 reverse=True)]
            #让FP树进行生长
            updateTree(orderedItems,retTree,headerTable,count)
    return retTree,headerTable

def updateTree(items,inTree,headerTable,count):
    """
    FP树的生长，开枝散叶
    :param items:元素项的list
    :param inTree:树
    :param headerTable:头指针
    :param count:事项出现次数
    :return:
    """
    #测试事务中第一个元素是否为子节点
    if items[0] in inTree.children:
        #若是，则更新计数
        inTree.children[items[0]].inc(count)
    else:
        #若不是，则创建一个新的treeNode作为树的子节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        #更新头指针
        if headerTable[items[0]][1] == None:
            #若为空，则指向自己
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            #若不为空，则需要指向新节点
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    #不断迭代调用自身，递归，直到所有元素遍历
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    """
    将节点链接指向树中该元素项的每个实例
    :param nodeToTest:节点链接
    :param targetNode:
    :return:
    """
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):
    """
    追溯FP树，收集叶节点之前的全部元素项
    :param leafNode: 叶节点
    :param prefixPath: 前缀路径，即该节点到树根的全部内容
    :return:
    """
    if leafNode.parent != None:
        #若叶子节点存在父节点，则添加父节点
        prefixPath.append(leafNode.name)
        #递归查找，直到树根
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    """
    查找条件模式基
    :param basePat:元素项
    :param treeNode:树节点（与HeadTable对应）
    :return:
    """
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
    遍历
    :param inTree:树
    :param headerTable:头指针
    :param minSup:最小支持度
    :param preFix:条件模式基
    :param freqItemList:频繁项集的集合
    :return:
    """
    #头指针表中元素进行从小到大地排序处理，记住此时的HeaderTable第一个元素存储元素项出现次数
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[0])]
    #遍历每个频繁项，且从最小的元素项开始
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        #生成条件模式基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)




def loadSimpDat():
    # 简单例子
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    # 主要采用frozenset
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict



def example():
    #第一个例子
    # rootNode = treeNode('pyramid',9,None)
    # rootNode.children['eye'] = treeNode('eye',13,None)
    # rootNode.disp()
    # #第二个例子
    # simpDat = loadSimpDat()
    # initSet = createInitSet(simpDat)
    # myFPtree,myHeaderTab = createTree(initSet,3)
    # myFPtree.disp()
    # #第三个例子
    # temp = findPrefixPath('x',myHeaderTab['x'][1])
    # print(temp)
    # temp = findPrefixPath('z', myHeaderTab['z'][1])
    # print(temp)
    # temp = findPrefixPath('r', myHeaderTab['r'][1])
    # print(temp)
    #第四个例子
    parsedDat = [line.split() for line in open('./machinelearninginaction3x-master/Ch12/kosarak.dat')]
    initSet = createInitSet(parsedDat)
    myFPtree,myHeaderTab = createTree(initSet,100000)
    myFreqList = []
    mineTree(myFPtree,myHeaderTab,100000,set([]),myFreqList)
    print(myFreqList)



if __name__ == '__main__':
    example()
