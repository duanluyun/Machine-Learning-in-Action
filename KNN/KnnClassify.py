from numpy import *
import operator
def loadDataSet():
    dataSet=array([
        [1.0,1.1],
        [1.0,1.0],
        [0,0],
        [0,0.1]
    ])

    labels=["A","A","B","B"]
    return dataSet,labels


def knnClassify(inX,dataSet,labels,k):
    m,n=shape(dataSet)
    diff=tile(inX,(m,1))-dataSet
    print(diff)
    seqDiff=diff**2
    print(seqDiff)
    seqDistance=sum(seqDiff,axis=1)
    print(seqDistance)
    distance=seqDistance**0.5
    print(distance)
    sortDistanceIndex=distance.argsort()
    print(sortDistanceIndex)
    classCount={}
    for i in range(k):
        currentLabel=labels[sortDistanceIndex[i]]
        if currentLabel not in classCount:
            classCount[currentLabel]=0
        classCount[currentLabel]+=1

    sortClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortClassCount)
    return sortClassCount[0][0]

if __name__=="__main__":
    dataSet,labels=loadDataSet()
    inX=[1,1]
    print(knnClassify(inX,dataSet,labels,2))
