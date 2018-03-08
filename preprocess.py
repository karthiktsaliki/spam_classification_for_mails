import glob
import random
import re
from collections import Counter
import numpy as np

def create_bag_of_words(filePaths):
    bagOfWords = []
    regex = re.compile("X-Spam.*\n")
    for filePath in filePaths:
        with open(filePath, encoding ="latin-1") as f:
            raw = f.read()
            raw = re.sub(regex,'',raw)
            tokens = raw.split()
            for token in tokens:
                bagOfWords.append(token)
    return bagOfWords

def get_feature_matrix(filePaths, featureDict):
    featureMatrix = np.zeros(shape=(len(filePaths),
                                      len(featureDict)),
                               dtype=float)
    regex = re.compile("X-Spam.*\n")
    for i,filePath in enumerate(filePaths):
        with open(filePath, encoding ="latin-1") as f:
            _raw = f.read()
            raw = re.sub(regex,'',_raw)
            tokens = raw.split()
            fileUniDist = Counter(tokens)
            for key,value in fileUniDist.items():
                if key in featureDict:
                    featureMatrix[i,featureDict[key]] = value
    return featureMatrix

def regularize_vectors(featureMatrix):
    for doc in range(featureMatrix.shape[0]):
        totalWords = np.sum(featureMatrix[doc,:],axis=0)
        featureMatrix[doc,:] = np.multiply(featureMatrix[doc,:],(1/(totalWords+1)))

    return featureMatrix

def input_data(hamDir,spamDir,percentTest):
    pathLabelPairs={}
    for hamPath in glob.glob(hamDir+'/*'):
        pathLabelPairs.update({hamPath:"0.,1."})
    for spamPath in glob.glob(spamDir+'/*'):
        pathLabelPairs.update({spamPath:"1.,0."})
    
    numTest = int(percentTest * len(pathLabelPairs))
    testing = set(random.sample(pathLabelPairs.items(),numTest))

    for entry in testing:
        del pathLabelPairs[entry[0]]
    
    trainPaths=[]
    trainY=[]
    for item in pathLabelPairs.items():
        trainPaths.append(item[0])
        trainY.append([float(i) for i in item[1].split(',')])
    del pathLabelPairs
    trainY=np.asarray(trainY)

    testPaths=[]
    testY=[]
    for item in testing:
        testPaths.append(item[0])
        testY.append([float(i) for i in item[1].split(',')])
    del testing
    testY=np.asarray(testY)
    
    bagOfWords = create_bag_of_words(trainPaths)

    k=5
    freqDist = Counter(bagOfWords)
    newBagOfWords=[]
    for word,freq in freqDist.items():
        if freq > k:
            newBagOfWords.append(word)
    features = set(newBagOfWords)
    featureDict = {feature:i for i,feature in enumerate(features)}

    trainX = get_feature_matrix(trainPaths,featureDict)
    testX = get_feature_matrix(testPaths,featureDict)
    
    trainX = regularize_vectors(trainX)
    testX = regularize_vectors(testX)

    return trainX, trainY, testX, testY
