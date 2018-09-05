from numpy import *
import re


def LoadDataSet():
    Nom=['My dog has flea problems,help please','My dalmation is so cute, I love him','Mr licks ate my steak, how to stop him']
    Abus=['maybe not take him to dog park stupid','stop posting stupid worthless garbage','quit buying worthless dog food stupid']
    return Nom ,Abus

def createVocabList():
    VocabSet=set([])
    Nom,Abus=LoadDataSet()
    for docs in (Nom,Abus):
        for line in docs:
            VocabSet=VocabSet|set(split(line))
    return list(VocabSet)


def bagOfWordsToVector(VocabSet,WordSet):
    returnVect=[0]*len(VocabSet)
    for i in WordSet:
        if i in VocabSet:
            returnVect[VocabSet.index(i)]+=1
        else:
            print('the %s word is not in my Vocabulary'%i)
    return returnVect


def split(string):
    regExp=re.compile(r"\W*")
    tokens=regExp.split(string)
    res=[i.lower() for i in tokens if len(i)>=1]
    return res



def trainNB(trainMatrix,trainCategory):
    pAbusive=sum(trainCategory)/float(len(trainCategory))
    p1Num=ones(len(trainMatrix[0]))
    p1Domn=2.0
    p0Num=ones(len(trainMatrix[0]))
    p0Domn=2.0
    for i in range(len(trainMatrix)):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Domn+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Domn+=sum(trainMatrix[i])
    p1vect=log(p1Num/p1Domn)
    p0vect=log(p0Num/p0Domn)
    return p1vect,p0vect,pAbusive


def classifyNB(vecToClassify,p1vect,p0vect,pAbusive):
    p1=sum(vecToClassify*p1vect)+log(pAbusive)
    p0=sum(vecToClassify*p0vect)+log(1-pAbusive)
    if p1>p0:
        return 1
    else:
        return 0


if __name__=="__main__":
    docToClassify=input()
    VocbList=createVocabList()
    trainMatrix=[]
    Nom,Abus=LoadDataSet()
    for docs in (Nom,Abus):
        for line in docs:
            trainMatrix.append(bagOfWordsToVector(VocbList,split(line)))
    categoryList=[0,0,0,1,1,1]
    p1vect,p0vect,pAbusive=trainNB(array(trainMatrix),array(categoryList))
    vecToClassify=bagOfWordsToVector(VocbList,split(docToClassify))
    print(classifyNB(array(vecToClassify),p1vect,p0vect,pAbusive))

