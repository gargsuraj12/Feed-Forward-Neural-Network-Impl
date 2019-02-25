import pandas as pd
import numpy as np
import random
from sklearn import metrics
import math

costThrash = 0.5
batchSize = None
learningRate = None

# Splits the raw data into training data set and test data set
def splitTrainTest(df, testSize):
    if isinstance(testSize, float):
        testSize = round(testSize*len(df))

    indices = df.index.tolist()
    randIndices = random.sample(population=indices, k=testSize)
    valData = df.loc[randIndices]
    trainData = df.drop(randIndices)
    return trainData, valData 

# def calcAccuracy(actualList, predictedList):
#     tp = 0
#     for i in range(len(actualList)):
#         if actualList[i] == predictedList[i]:
#             tp += 1
#     return (tp/len(actualList))*100         


def calcAccuracy(actualList, predictedList):
    confusion_matrix = metrics.cluster.contingency_matrix(actualList, predictedList)
    print("Confusion Matrix:\n",confusion_matrix)
    return (np.sum(np.amax(confusion_matrix, axis=0))/np.sum(confusion_matrix))*100

class Layer:
    def __init__(self, weights, bias):
        # Hyperparameters for each layer in NeuralNetwork
        self.weights = weights
        self.bias = bias
        self.z = None
        self.activation = None
        self.delta = 0

# To represent neural network
class NeuralNetwork:
    def __init__(self):
        self.layersCount = None
        self.layerList = []
        self.actual = None
        self.error = None

    # Function to initialize and assign random weights and bias to the layers
    def buildNetwork(self, layerNodes):
        self.layersCount = len(layerNodes)
        # Initializing all the layers except the last one with random weights and bias
        for i in range(self.layersCount-1):
            weights = np.random.randn(layerNodes[i], layerNodes[i+1])*np.sqrt(1/layerNodes[i])
            bias = np.random.randn(1, layerNodes[i+1])
            layerObj = Layer(weights, bias)
            self.layerList.append(layerObj)
            del layerObj

        # Initializing empty output list
        layerObj = Layer(None, None)
        self.layerList.append(layerObj)

    def sigmoid(self, z):
        return np.nan_to_num(1/(1 + np.exp(-z)))

    def sigmoidDerivative(self, z):
        # return np.exp(-z)/((1+np.exp(-z))**2)
        # return self.sigmoid(z) * (1-self.sigmoid(z))    
        return np.multiply(z, (1-z))

    def tanH(self, z):
        result = 2.0/(1.0 + np.exp(-(2*z))) - 1
        return result
    
    def tanHDerivative(self, z):
        result = 1 - (self.tanH(z)**2)
        return result

    # This function converts the actual label in the matrix form
    def setActualAsMatrix(self, trainLabel, outputNodes):
        # x = trainLabel.shape[0]
        # a = []
        # print(self.actual)
        # for i in range(x):
        #     row = [0]*outputNodes
        #     j = trainLabel[i]
        #     row[j] = 1
        #     # print(row)
        #     a.append(row)
        # self.actual = np.asarray(a)

        targets = trainLabel.reshape(-1)
        one_hot_targets = np.eye(outputNodes)[targets]
        # print(one_hot_targets)
        # print(one_hot_targets[:,0])
        self.actual = one_hot_targets

    # Propogating forward with prevoiusly established weights and biases
    def forwardPropagation(self, trainData):
        self.layerList[0].activation = trainData
        for i in range(1, self.layersCount):
            prevActivation = self.layerList[i-1].activation
            prevWeights = self.layerList[i-1].weights
            prevBias = self.layerList[i-1].bias

            z = np.dot(prevActivation, prevWeights)
            z = z + prevBias
            self.layerList[i].z = z
            # z = np.asarray(z)
            # print(np.unique(z))
            # input()
            currActivation = self.sigmoid(z)
            # input()
            # currActivation = self.tanH(z)
            self.layerList[i].activation = currActivation

        return self.layerList[-1].activation

    # Normalizing output layer's activation such that sum of each row will be 1
    def normalizeOutput(self):
        # Normalizing output layer's activation
        mat = self.layerList[-1].activation.tolist()
        activation = []
        for row in mat:
            row = np.asarray(row)
            activation.append(row/np.sum(row))
        self.layerList[-1].activation = np.asarray(activation)

    def crossEntropyDerivative(self):
        y = self.actual
        a = self.layerList[-1].activation
        err = []
        for i in range(y.shape[0]):
            row = []
            for j in range(y.shape[1]):
                t = -((y[i][j]/a[i][j])-((1-y[i][j])/(1-a[i][j])))
                row.append(t)
            err.append(row)
        self.error = np.nan_to_num(np.asarray(err))

    def mseDerivative(self,m):
        self.error = (self.layerList[-1].activation - self.actual)/m

    def backPropagation(self, trainLabel):
        # currZ = self.layerList[-1].z
        currActivation = self.layerList[-1].activation
        self.layerList[-1].delta = np.multiply(self.error, self.sigmoidDerivative(currActivation))
        # self.layerList[-1].delta = np.multiply(self.error, self.tanHDerivative(currZ))

        # Running loop from second last layer for calculating theta's for every hidden layer
        for i in range(self.layersCount-2, 0, -1):
            nextLayerDelta = self.layerList[i+1].delta
            # currWeights = self.layerList[i].weights
            currWeights = self.layerList[i].weights
            # currZ = self.layerList[i].z
            currActivation = self.layerList[i].activation
            deltaDotWeights = np.dot(nextLayerDelta, currWeights.T)
            delta = np.multiply(deltaDotWeights, self.sigmoidDerivative(currActivation))
            # delta = np.multiply(deltaDotWeights, self.tanHDerivative(currZ))
            self.layerList[i].delta = delta
        
        # Calculating gradient and updating weights and bias at each layer except output layer
        # for i in range(self.layersCount-1):
        for i in range(self.layersCount-2, -1, -1):
            # print("Delta for layer:",i+1, " is: ",self.layerList[i].delta)
            nextDelta = self.layerList[i+1].delta
            currActivation = self.layerList[i].activation
            weightGradient = np.dot(currActivation.T, nextDelta)
            biasGradient = np.ones((1,nextDelta.shape[0]))
            biasGradient = np.dot(biasGradient, nextDelta)
            self.layerList[i].weights = self.layerList[i].weights - (learningRate/batchSize)*weightGradient
            self.layerList[i].bias = self.layerList[i].bias - (learningRate/batchSize)*biasGradient
            # print("New weights for layer ", i+1, " are: ", self.layerList[i].weights)


    # Calling various functions in appropriate order
    def trainNetwork(self, trainData, trainLabel,outputNodes):
        self.setActualAsMatrix(trainLabel,outputNodes)
        self.forwardPropagation(trainData)
        # self.normalizeOutput()
        self.crossEntropyDerivative()
        # self.mseDerivative(trainData.shape[0])
        self.backPropagation(trainLabel)

    def predict(self, valData):
        predicted = []
        for row in valData:
            activation = self.forwardPropagation(row)
            # print("Activation is: ", activation)
            predicted.append(np.argmax(activation))
        return predicted    


def meanSquareError(actualLabel, predictedLabel):
    n = len(actualLabel)
    error = 0
    for i in range(n):
        error += pow((actualLabel[i]-predictedLabel[i]),2)

    error = error/n
    return error

if __name__ == "__main__":
    df = pd.read_csv("apparel-trainval.csv")
    trainData, valData = splitTrainTest(df, 0.2)
    
    trainLabel = trainData["label"].values
    valLabel = valData["label"].values
    
    trainData = trainData.drop(["label"], axis=1)
    valData = valData.drop(["label"], axis=1)
    
    # mean,std = trainData.mean(),trainData.std()
    # trainData = (trainData-mean)/std
    # valData = (valData-mean)/std

    # df = pd.read_csv("Iris.csv", names=["C1", "C2", "C3", "C4", "C5"])
    # trainData, valData = splitTrainTest(df, 0.2)

    # trainLabel = trainData["C5"].values
    # valLabel = valData["C5"].values
    
    # trainData = trainData.drop(["C5"], axis=1)
    # valData = valData.drop(["C5"], axis=1)

    # Converting training and validation data to numpy
    trainData = trainData.values
    valData = valData.values

    train_sd = np.std(trainData)
    train_mean = np.mean(trainData)
    
    trainData = (trainData - train_mean)/train_sd
    valData = (valData - train_mean)/train_sd

    inputNodes = trainData.shape[1]
    outputNodes = 10
    # This list represents the number of layers and number of nodes in each layer
    layerNodes = [inputNodes, 100, 100, outputNodes] #16, 
    # layerNodes = [inputNodes, 4, outputNodes] # For iris-dataset 
    
    nn = NeuralNetwork()
    nn.buildNetwork(layerNodes)
    batchSize = 100
    learningRate = 0.1
    count = 1
    iterations = 50
    print(type(trainLabel))
    for k in range(iterations):
        print("Iteration: ",k+1)
        start = 0
        end = batchSize
        numBatches = math.floor(trainData.shape[0]/batchSize)
        for i in range(numBatches):
            # if len(data) < batchSize:
            #     data.append(trainData[i])
            #     label.append(trainLabel[i])
            # else:
                # data = np.asarray(data)
                # label = np.asarray(label)
            data = trainData[start:end, :]
            label = trainLabel[start:end]
            nn.trainNetwork(data, label, outputNodes)
            # print("Training completed for batch no:", count)
            count += 1
            start += batchSize
            end += batchSize
                # data = []
                # label = []

    predicted = nn.predict(valData)
    # print("Predicted: ", predicted)
    print("Accuracy is: ", calcAccuracy(valLabel.flatten().tolist(), predicted))
	# Hello
    
