#!/user/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import feedparser

def loadDataSet():
    #实验样本
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #样本分类
    classVec = [0,1,0,1,0,1]    #1 代表侮辱文件，0 代表正常言论
    return postingList,classVec

def createVocabList(dataSet):
    """
    单词汇总表
    :param dataSet:
    :return:
    """
    vocabSet = set([])
    for document in dataSet:
        # union of the two sets
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """词集模型
    文本转换为数值向量
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" %word)
    return returnVec

def bagOfWords2VecMN(vocabList,inputSet):
    """词袋模型

    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix,trainCategory):
    """
    训练样本，计算每个单词的条件概率
    :param trainMatrix:
    :param trainCategory:
    :return:
    """
    numTrainDocs = len(trainMatrix)     #样本个数
    numWords = len(trainMatrix[0])      #词汇总的单词个数
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # p0Num = np.zeros(numWords)    考虑到概率为0情况需要修改，采用Laplace平滑
    # p1Num = np.zeros(numWords)
    # p0Denom = 0.0
    # p1Denom = 0.0
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vect = p1Num/p1Denom    考虑到程序下溢，采取对数
    # p0Vect = p0Num/p0Denom
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    输入样本，进行分类
    :param vec2Classify:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    """
    test检验
    :return:
    """
    listOPosts,listClasses = loadDataSet()
    myVocalList = createVocabList(listOPosts)
    trainMat = []
    for postingDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocalList,postingDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEnry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocalList,testEnry))
    print(testEnry,' calssified as: ', classifyNB(thisDoc,p0V,
                                                   p1V,pAb))
    testEnry = ['i', 'love']
    thisDoc = np.array(setOfWords2Vec(myVocalList,testEnry))
    print(testEnry, ' calssified as: ', classifyNB(thisDoc, p0V,
                                                   p1V, pAb))

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        worldList = textParse(open(r'machinelearninginaction3x-master'
                                   r'\Ch04\email\spam\%d.txt' %i,
                                   encoding='ISO-8859-1' ).read())
        docList.append(worldList)
        fullText.extend(worldList)
        classList.append(1)
        worldList = textParse(open(r'machinelearninginaction3x-master'
                                   r'\Ch04\email\ham\%d.txt' %i,
                                   encoding='ISO-8859-1' ).read())
        docList.append(worldList)
        fullText.extend(worldList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
    #随机构建训练集
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
    trainingMat = []
    trainingClasses = []
    for docIndex in trainingSet:
        trainingMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainingClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(trainingMat,trainingClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if (classifyNB(np.array(wordVector),p0V,p1V,pSpam) !=
            classList[docIndex]):
            errorCount += 1
    print("the error rate is: %f" %float(errorCount/len(testSet)))
    return float(errorCount/len(testSet))


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen)
    testSet = []           #create test set
    for i in range(20):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V
























if __name__ == '__main__':
    ny = feedparser.parse("http://newyork.craigslist.org/stp/index.rss")
    sf = feedparser.parse("http://sfbay.craigslist.org/stp/index.rss")
    localWords(ny,sf)