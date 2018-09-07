from numpy import *
import matplotlib.pyplot as plt

def loadData():
    dataMat=[]
    labelMat=[]
    File=open('/home/sam/Downloads/machinelearninginaction/Ch05/testSet.txt')
    for line in File.readlines():
        splits=line.strip().split()
        dataMat.append([1.0,float(splits[0]),float(splits[1])])
        labelMat.append(int(splits[2]))
    return dataMat,labelMat

def sigMod(x):
    return longfloat(1.0/(1+exp(-x)))


def gradeAscent(dataMat,labelMat):

    dataMat=mat(dataMat)
    labelMat=mat(labelMat).transpose()
    m, n = shape(dataMat)
    weights = ones((n, 1))
    maxCycle=500
    alpha=0.001
    for i in range(maxCycle):
        H=sigMod(dataMat*weights)
        error=labelMat-H
        weights=weights+alpha*dataMat.transpose()*error
    return array(weights)

def randomGradeAscentII(dataMat,labelMat):
    m,n=shape(dataMat)
    weights=ones(n)
    maxCycle=500
    for i in range(maxCycle):
        for j in range(m):
            alpha=4.0/(i+j+1.0)+0.01
            nextIndex=int(random.uniform(0,len(dataMat)))
            s=sum(dataMat[nextIndex]*weights)
            H=sigMod(sum(dataMat[nextIndex]*weights))
            error=labelMat[nextIndex]-H
            weights=weights+alpha*error*dataMat[nextIndex]
    return weights


def randomGradeAscentI(dataMat,labelMat):
    m,n=shape(dataMat)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        H=sigMod(sum(dataMat[i]*weights))
        error=labelMat[i]-H
        weights=weights+alpha*error*dataMat[i]
    return weights

def plotBestFit(weights):
    dataMat,labelsMat=loadData()
    dataMat=array(dataMat)
    m,n=shape(dataMat)
    Ax=[]
    Ay=[]
    Bx=[]
    By=[]
    for i in range(m):
        if int(labels[i])==1:
            Ax.append(dataMat[i][1])
            Ay.append(dataMat[i][2])
        else:
            Bx.append(dataMat[i][1])
            By.append(dataMat[i][2])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(Ax,Ay,s=30,c='red',marker='s')
    ax.scatter(Bx,By,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]

    ax.plot(x,y)
    plt.xlabel('x1')
    plt.xlabel('x2')
    plt.show()


if __name__=='__main__':
    dataSet,labels=loadData()
    weights=gradeAscent(dataSet,labels)
    print(weights)
    plotBestFit(weights)
    weights1=randomGradeAscentII(array(dataSet),labels)
    plotBestFit(weights1)
    weights2=randomGradeAscentI(array(dataSet),labels)
    plotBestFit(weights2)





