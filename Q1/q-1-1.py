import pandas as pd
import numpy as np
import random
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import pickle, os

batchSize = None
learningRate = None
SIGMOID = "Sigmoid"
TANH = "TanH"
RELU = "RelU"
LOADFROMFILE = 4
TRAINNETWORK = 5

# Splits the raw data into training data set and test data set
def splitTrainTest(df, testSize):
    if isinstance(testSize, float):
        testSize = round(testSize*len(df))

    indices = df.index.tolist()
    randIndices = random.sample(population=indices, k=testSize)
    valData = df.loc[randIndices]
    trainData = df.drop(randIndices)
    return trainData, valData        

def calcAccuracy(actualList, predictedList):
    confusion_matrix = metrics.cluster.contingency_matrix(actualList, predictedList)
    print("Confusion Matrix:\n",confusion_matrix)
    return (np.sum(np.amax(confusion_matrix, axis=0))/np.sum(confusion_matrix))*100

def oneHotEncoding(trainLabel, outputNodes):
    trainLabel = np.asarray(trainLabel)
    targets = trainLabel.reshape(-1)
    one_hot_targets = np.eye(outputNodes)[targets]
    return np.asarray(one_hot_targets)

def crossEntropyError(yActual, yCap):
    error = -np.sum(yActual*np.log(yCap+1e-9))/(yActual.shape[0])
    return error


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
        self.funcType = None

    # Function to initialize and assign random weights and bias to the layers
    def buildNetwork(self, layerNodes):
        self.layersCount = len(layerNodes)
        # Initializing all the layers except the last one with random weights and bias
        for i in range(self.layersCount-1):
            weights = np.random.randn(layerNodes[i], layerNodes[i+1])*np.sqrt(1/layerNodes[i])
            # weights = np.random.randn(layerNodes[i], layerNodes[i+1])*(0.1)
            bias = np.random.randn(1, layerNodes[i+1])
            layerObj = Layer(weights, bias)
            self.layerList.append(layerObj)
            del layerObj

        # Initializing empty output list
        layerObj = Layer(None, None)
        self.layerList.append(layerObj)

    # Various activation functions and their derivatives
    def sigmoid(self, z):
        return np.nan_to_num(1/(1 + np.exp(-z)))

    def sigmoidDerivative(self, z):  
        return np.multiply(z, (1-z))

    def tanH(self, z):
        return np.tanh(z)
        # return np.nan_to_num(2.0/(1.0 + np.exp(-(2*z))) - 1)
    
    def tanHDerivative(self, z):
        return (1 - np.tanh(z)**2)

    def relU(self, z):
        return np.maximum(z,0)

    def relUDerivative(self, z):
        z[z<=0] = 0
        z[z>0] = 1
        return z

    def softmax(self, a):
        z = np.exp(a)
        zSum = np.sum(z, axis=1)
        for i in range(len(z)):
            z[i] = z[i]/zSum[i]
        return z

    def softmaxDerivative(self, z):
        x = self.softmax(z)
        return x*(1-x)
    
    # This function converts the actual label in the matrix form
    def oneHotEncoding(self, trainLabel, outputNodes):
        targets = trainLabel.reshape(-1)
        one_hot_targets = np.eye(outputNodes)[targets]
        self.actual = one_hot_targets

    # Normalizing output layer's activation such that sum of each row will be 1
    def normalizeOutput(self):
        # Normalizing output layer's activation
        mat = self.layerList[-1].activation.tolist()
        activation = []
        for row in mat:
            row = np.asarray(row)
            activation.append(row/np.sum(row))
        self.layerList[-1].activation = np.asarray(activation)

    def crossEntropyError(self, y, yCap):
        loss = 0
        for i in range(len(y)):
            x = yCap[i,y[i]]
            loss += -(np.log(x))
        return loss/len(y)

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

    # Needs to be corrected
    def mseDerivative(self,m):
        self.error = (self.layerList[-1].activation - self.actual)/m


    # Propogating forward with prevoiusly established weights and biases
    def forwardPropagation(self, trainData):
        self.layerList[0].activation = trainData
        for i in range(1, self.layersCount):
            prevActivation = self.layerList[i-1].activation
            prevWeights = self.layerList[i-1].weights
            prevBias = self.layerList[i-1].bias

            z = np.dot(prevActivation, prevWeights)
            z = z + prevBias
            # self.layerList[i].z = z   # Not neccessary

            # For output layer
            if i == self.layersCount-1:
                currActivation = self.sigmoid(z)
                # currActivation = self.softmax(z)
            # For other layers
            else:    
                if self.funcType == SIGMOID:
                    currActivation = self.sigmoid(z)
                elif self.funcType == TANH:
                    currActivation = self.tanH(z)
                else:
                    currActivation = self.relU(z)    
            self.layerList[i].activation = currActivation

        return self.layerList[-1].activation


    # Propogating backwards to calculates deltas and updating weights and biases
    def backPropagation(self, trainLabel):
        currActivation = self.layerList[-1].activation
        self.layerList[-1].delta = np.multiply(self.error, self.sigmoidDerivative(currActivation))
        # self.layerList[-1].delta = np.multiply(self.error, self.softmaxDerivative(currActivation))

        # Running loop from second last layer for calculating theta's for every hidden layer
        for i in range(self.layersCount-2, 0, -1):
            nextLayerDelta = self.layerList[i+1].delta
            currWeights = self.layerList[i].weights
            # currZ = self.layerList[i].z
            deltaDotWeights = np.dot(nextLayerDelta, currWeights.T)
            
            currActivation = self.layerList[i].activation
            if self.funcType == SIGMOID:
                activationDerivative = self.sigmoidDerivative(currActivation)
            elif self.funcType == TANH:
                activationDerivative = self.tanHDerivative(currActivation)
            else:
                activationDerivative = self.relUDerivative(currActivation)

            delta = np.multiply(deltaDotWeights, activationDerivative)
            # delta = np.multiply(deltaDotWeights, self.tanHDerivative(currZ))
            self.layerList[i].delta = delta
        
        # Calculating gradient and updating weights and bias at each layer except output layer
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
        self.oneHotEncoding(trainLabel,outputNodes)
        self.forwardPropagation(trainData)
        # self.normalizeOutput()
        error = self.crossEntropyError(trainLabel, self.layerList[-1].activation)
        self.crossEntropyDerivative()
        # self.mseDerivative(trainData.shape[0])
        self.backPropagation(trainLabel)
        return error

    def predict(self, valData):
        predicted = []
        self.kuchbhi = None
        index = 0
        for row in valData:
            activation = self.forwardPropagation(row)
            if index == 0:
                self.kuchbhi = activation
            else:    
                np.concatenate((self.kuchbhi, activation))
            predicted.append(np.argmax(activation))
            index += 1
        return predicted    


def plotLossVsIterations(iterationsList, trainLossList):
    plt.plot(iterationsList, trainLossList)
    plt.xlabel("Iterations")
    plt.ylabel("Loss %")
    plt.title("Iterations Vs Loss")
    plt.savefig("q-1-1_IterationsVsLoss.png")
    plt.close()

def plotEpochsVsAccuracy(epochList, trainAccuracyList, valAccuracyList):
    plt.plot(epochList, trainAccuracyList)
    plt.plot(epochList, valAccuracyList)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Epochs Vs Accuracy")
    plt.legend(["TrainData", "Validation Data"], loc = "lower right")
    plt.savefig("q-1-1_Epochs_Vs_Accuracy.png")
    plt.close()

def plotHiddenLayersVsAccuracy(hiddenLayerList, valAccuracyList):
    plt.plot(hiddenLayerList, valAccuracyList)
    plt.xlabel("Hidden Layers")
    plt.ylabel("Accuracy")
    plt.title("Hidden Layers Vs Validation Data Accuracy")
    plt.savefig("q-1-1_HiddenLayers_Vs_ValAccuracy.png")
    plt.close()

def plotHiddenLayersVsPredictionError(hiddenLayerList, trainLossList, valLossList):
    plt.plot(hiddenLayerList, trainLossList)
    plt.plot(hiddenLayerList, valLossList)
    plt.xlabel("Hidden Layers")
    plt.ylabel("PredictionError")
    plt.title("Hidden Layers Vs PredictionError")
    plt.legend(["TrainData", "Validation Data"], loc = "lower right")
    plt.savefig("q-1-1_HiddenLayers_Vs_PredictionError.png")
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv("apparel-trainval.csv")
    trainData, valData = splitTrainTest(df, 0.2)
    
    trainLabel = trainData["label"].values
    valLabel = valData["label"].values
    
    trainData = trainData.drop(["label"], axis=1)
    valData = valData.drop(["label"], axis=1)
    
    # Converting training and validation data to numpy
    trainData = trainData.values
    valData = valData.values

    # Normalizing the data
    trainSD = np.std(trainData)
    trainMean = np.mean(trainData)
    trainData = (trainData - trainMean)/trainSD
    valData = (valData - trainMean)/trainSD

    mode = TRAINNETWORK
    # mode = LOADFROMFILE
    parametersFile = None

    if mode == LOADFROMFILE:
        try:
            parametersFile = open("parameters", "rb")
            nn = pickle.load(parametersFile)
        except:
            print("Error while loading weights and bias from the file..")
        finally:
            parametersFile.close()
    
    else:
        inputNodes = trainData.shape[1]
        outputNodes = 10
        # This list represents the number of layers and number of nodes in each layer
        layerNodes = [inputNodes, outputNodes]
        batchSize = 100
        learningRate = 0.1
        valAccuracyList = []
        trainAccuracyList = []
        trainLossList = []
        valLossList = []
        # epochList = [5, 10, 15, 20, 25, 30]
        epochList = [20]
        nn = None
        iterations = 10
        hiddenLayerList = []
        # for iterations in epochList:
        #     layerNodes = [inputNodes, 50, 50, outputNodes]
        for hNum in range(1,3):

            layerNodes = layerNodes[:-1]
            layerNodes.append(50)
            layerNodes.append(outputNodes)
            hiddenLayerList.append(hNum)
            print("Trainig neural network with Hidden layers:", hNum)
            # print("For epoch:", iterations)
            nn = NeuralNetwork()
            nn.funcType = SIGMOID    
            nn.buildNetwork(layerNodes)
            # print("Trainig neural network with function type: ", nn.funcType)
            iterationsList = []

            for k in range(iterations):
                print("Iteration: ",k+1)
                iterationsList.append(k+1)
                start = 0
                end = batchSize
                loss = 0
                numBatches = math.ceil(trainData.shape[0]/batchSize)
                for i in range(numBatches):
                    data = trainData[start:end, :]
                    label = trainLabel[start:end]
                    loss += nn.trainNetwork(data, label, outputNodes)
                    start += batchSize
                    end += batchSize
                
                # loss = loss/trainData.shape[0]   
                # trainLossList.append(loss)
                # print("Loss: ", loss)

            # valLabel = valLabel.flatten().tolist()
            predicted = []
            # Prediction on validation data
            predicted = nn.predict(valData)
            valAccuracy = calcAccuracy(valLabel.flatten().tolist(), predicted)
            print("Accuracy on validation data is: ", valAccuracy)
            valAccuracyList.append(valAccuracy)
            valActual = oneHotEncoding(valLabel, outputNodes)
            # valCap = oneHotEncoding(predicted, outputNodes)

            loss = crossEntropyError(valActual, nn.kuchbhi)
            valLossList.append(loss)

            predicted = nn.predict(trainData)
            trainAccuracy = calcAccuracy(trainLabel.flatten().tolist(), predicted)
            print("Accuracy on train data is: ", trainAccuracy)
            trainAccuracyList.append(trainAccuracy)
            trainActual = oneHotEncoding(trainLabel, outputNodes)
            # trainCap = oneHotEncoding(predicted, outputNodes)
            loss = crossEntropyError(trainActual, nn.kuchbhi)
            trainLossList.append(loss)

        if mode == TRAINNETWORK:
            print("Network Successfully trained..")
            # plotLossVsIterations(trainLossList, iterationsList)
            # plotEpochsVsAccuracy(epochList, trainAccuracyList, valAccuracyList)
            # print("ValLossList: ",valLossList)
            # print("trainLossList: ",trainLossList)
            # plotHiddenLayersVsAccuracy(hiddenLayerList, valAccuracyList)
            plotHiddenLayersVsPredictionError(hiddenLayerList, trainLossList, valLossList)

            

    if mode == LOADFROMFILE:
        predicted = []
        # Prediction on validation data
        predicted = nn.predict(valData)
        valAccuracy = calcAccuracy(valLabel.flatten().tolist(), predicted)
        print("Accuracy on validation data is: ", valAccuracy)

    # Prediction on test data
    testData = pd.read_csv("apparel-test.csv").values
    # Normalizing the test data
    testData = (testData - trainMean)/trainSD
    predicted = nn.predict(testData)
    np.savetxt('2018202003_prediction.csv', predicted, delimiter=',', fmt="%d")
    print("Test data prediction successfully saved to '2018202003_prediction.csv'")

    # Dumping the current neural network class object into the file
    try:
        parametersFile = open("parameters", "wb")
        pickle.dump(nn, parametersFile)
    except:
        print("Error while dumping neural network parametrs into pickle file")
    finally:        
        parametersFile.close()
