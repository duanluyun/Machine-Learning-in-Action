from math import log
import operator

def calculateEntropy(DataSet):
    numEntries=len(DataSet)
    labelCounter={}
    for feaVec in DataSet:
        currentLabel=feaVec[-1]
        if currentLabel not in labelCounter:
            labelCounter[currentLabel]=0
        labelCounter[currentLabel]+=1
    Entropy=0.0
    for key in labelCounter.keys():
        probe=labelCounter[key]/float(numEntries)
        Entropy-=probe*log(probe,2)
    return Entropy

def splitDataSetByFeat(DataSet,axis,value):
    subSet=[]
    for featVec in DataSet:
        if featVec[axis]==value:
            reduceFeatVect=featVec[:axis]
            reduceFeatVect.extend(featVec[axis+1:])
            subSet.append(reduceFeatVect)
    return subSet

def findBestFeatureToSplitDataset(DataSet):
    numOfFeatures=len(DataSet[0])-1
    bestInfoGain=0.0
    baseEntropy=calculateEntropy(DataSet)
    bestFeat=-1
    for i in range(numOfFeatures):
        feaValList=[example[i] for example in DataSet]
        uniqueFeaVal=set(feaValList)
        tempEntropy=0.0
        for value in uniqueFeaVal:
            subSet=splitDataSetByFeat(DataSet,i,value)
            prob=len(subSet)/float(len(DataSet))
            tempEntropy+=prob*calculateEntropy(subSet)
        infoGain=baseEntropy-tempEntropy
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeat=i
    return bestFeat

def MarjorityCounter(classList):
    classCounter={}
    for i in classList:
        if i not in classCounter:
            classCounter[i]=0
        classCounter[i]+=1
    sortedClassCounter=sorted(classCounter.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCounter[0][0]


def creatTree(DataSet,labels):
    classList=[example[-1] for example in DataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(DataSet[0])==1:
        return MarjorityCounter(classList)
    bestFeature=findBestFeatureToSplitDataset(DataSet)
    bestFeatureLabel=labels[bestFeature]
    myTree={bestFeatureLabel:{}}
    feaValList=[example[bestFeature] for example in DataSet]
    uniqueValList=set(feaValList)
    del(labels[bestFeature])
    for value in uniqueValList:
        subLabels=labels[:]
        myTree[bestFeatureLabel][value]=creatTree(splitDataSetByFeat(DataSet,bestFeature,value),subLabels)
    return myTree




def createDataSet():
    dataSet=[
        [1,1,'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels=['no surfacing','flippers']
    return dataSet,labels


if __name__=="__main__":
    dataSet,labels=createDataSet()
    print(creatTree(dataSet,labels))