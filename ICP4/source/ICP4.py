from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import numpy as np


def avg(lis):
    '''
    returns the average of all of the elements in a list
    '''
    if len(lis) == 0:
        return 0
    return sum(lis)/len(lis)

def getSummary(confusionMatrix):
    '''
    Calculates the accuracy, precision, recall, f-1 score for all classes
    Returns the average of these for all classes in a confusion matrix
    '''
    
    ACC = []
    PRE = []
    REC = []
    FOS = []
    
    classes = len(set(iris.target))
    totData = iris.data.shape[0]
    
    for i in range(classes):
        #calculate true positive
        tpNum = confusionMatrix[i][i]

        #calculate false negative
        fnNum = confusionMatrix[i].sum()-tpNum

        #calculate false positive
        fpNum = 0
        for j in range(classes):
            fpNum += confusionMatrix[j][i]
        fpNum = fpNum - tpNum

        #calculate true negative
        tnNum = totData - tpNum - fpNum - fnNum

        #calculate accuracy, precision, recall, f-1 score for a single class
        acc = (tpNum + tnNum)/totData
        pre = tpNum/(tpNum+fpNum)
        rec = tpNum/(fnNum+tpNum)
        fos = 2*((pre*rec)/(pre+rec))
        
        ACC.append(acc)
        PRE.append(pre)
        REC.append(rec)
        FOS.append(fos)

    #return the average accuracy, precision, recall, f-1 score.    
    accuracy = avg(ACC)
    precision = avg(PRE)
    recall = avg(REC)
    fonescore = avg(FOS)

    return (accuracy,precision,recall,fonescore)

#load dataset
iris = datasets.load_iris()

cont = True
while cont:
    try:
        k = int(input("How many splits for cross validation?"))

        #initialize arrays for accuracy, precision, recall, and f1score.
        GNBres = np.zeros(4)
        SVMres = np.zeros(4)
        
        for i in range(k):
            #split the training and test set
            xTrain, xTest, yTrain, yTest = train_test_split(iris.data,iris.target, test_size=0.4)

            #fit and test the Gaussian Naive Bayes classifier
            gnb = GaussianNB()
            NBPred = gnb.fit(xTrain, yTrain).predict(xTest)

            #fit and test the SVM classifier
            clf = svm.SVC(gamma='scale')
            SVMPred = clf.fit(xTrain, yTrain).predict(xTest)

            #get the confusion matrix and calculate the accuracy, precision, recall, f1score for all of the classes in the GNB model
            GNBconfusionMatrix = confusion_matrix(yTest, NBPred)
            GNBres+=np.array(getSummary(GNBconfusionMatrix)) #add for the avg later on 

            #get the confusion matrix and calculate the accuracy, precision, recall, f1score for all of the classes in the SVM model
            SVMconfusionMatrix = confusion_matrix(yTest, SVMPred)
            SVMres+=np.array(getSummary(SVMconfusionMatrix)) #add for the avg later on 

        #get average by dividing by the number of splits
        GNBres/=k
        SVMres/=k        

        #print results
        print("Gausian Naive Bayes:\nAccuracy: {0:.2f}%\nPrecision: {1:.2f}%\nRecall: {2:.2f}%\nF-1 Score: {3:.2f}%\n\n".format(GNBres[0]*100, GNBres[1]*100, GNBres[2]*100, GNBres[3]*100)) #see getSummary function for which value is which

        print("Support Vector Classification:\nAccuracy: {0:.2f}%\nPrecision: {1:.2f}%\nRecall: {2:.2f}%\nF-1 Score: {3:.2f}%".format(SVMres[0]*100, SVMres[1]*100, SVMres[2]*100, SVMres[3]*100)) #see getSummary function for which value is which

    except Exception as e: #pretty much here just in case the first input isn't an integer
        print(e)
        
    finally:
        choice = input("Press y to do it again.\n\n")
        if choice.lower() != 'y':
            cont = False


