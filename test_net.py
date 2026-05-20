import ctypes
import json
from enum import IntEnum
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

#ctypes preparations.
class cActivationFunctionType(IntEnum):
    sigmoid = 0
    tanhyp = 1
    ReLU = 2
    linear = 3
    softmax = 4

class cCostFunctionType(IntEnum):
    square = 0
    crossEntropy = 1
    logLikehood = 2

class cNeuron(ctypes.Structure):
    _fields_ = [
            ('layer', ctypes.c_int),
            ('index', ctypes.c_int),
            ('weightsCount', ctypes.c_int),
            ('aft', ctypes.c_int)
    ]

class cNeuralNetwork(ctypes.Structure):
    _fields_ = [
            ('net', ctypes.POINTER(cNeuron)),
            ('neuronsCount', ctypes.c_int),
            ('inputLayerNeuronsCount', ctypes.c_int),
            ('outputLayerNeuronsCount', ctypes.c_int),
            ('hiddenLayersCount', ctypes.c_int),
            ('neuronsPerHiddenLayer', ctypes.c_int),
            ('cft', ctypes.c_int),
            ('batchSize', ctypes.c_int),
            ('baseLearningRate', ctypes.c_double),
            ('l1RegularizationParameter', ctypes.c_double),
            ('l2RegularizationParameter', ctypes.c_double),
            ('decoupledWeightDecay', ctypes.c_double),
            ('noImprovementsEpochsLimit', ctypes.c_int),
            ('learningRateCurrentDecreaser', ctypes.c_double),
            ('learningRateDecreaserLimit', ctypes.c_double),
            ('beta1', ctypes.c_double),
            ('beta2', ctypes.c_double),
            ('eps', ctypes.c_double),
            ('biasCorrectionInAdamOptimizer', ctypes.c_bool),
            ('lastLayerFirstIndex', ctypes.c_int),
            ('weightsNumber', ctypes.c_int),
            ('biasesNumber', ctypes.c_int),
            ('resData', ctypes.POINTER(ctypes.c_double)),
            ('deltasData', ctypes.POINTER(ctypes.c_double)),
            ('weights', ctypes.POINTER(ctypes.c_double)),
            ('biases', ctypes.POINTER(ctypes.c_double)),
            ('iterationsOfWeightsUpdatesDone', ctypes.POINTER(ctypes.c_int)),
            ('weightsGradientsMovingAverages', ctypes.POINTER(ctypes.c_double)),
            ('biasesGradientsMovingAverages', ctypes.POINTER(ctypes.c_double)),
            ('weightsGradientSquaresMovingAverages', ctypes.POINTER(ctypes.c_double)),
            ('biasesGradientSquaresMovingAverages', ctypes.POINTER(ctypes.c_double))
    ]

netLib = ctypes.CDLL(os.getcwd() + '/nn_net.so')

netLib.createNetwork.restype = ctypes.POINTER(cNeuralNetwork)
netLib.testNetworkByEvalData.restype = ctypes.c_int
netLib.testNetworkByEvalData.argtypes = [cNeuralNetwork, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]

#Load data from MNIST library.
def loadMNIST(fileName):
    with open(fileName, "rb") as file:
        mainInfoBuffer = file.read(4)
        #print(mainInfoBuffer)
        dimensionsAmount = mainInfoBuffer[3]
        #print(dimensionsAmount)
        if dimensionsAmount > 0:
            sampleSize = 1
            dimensions = []
            for i in range(dimensionsAmount):
                dimensionBytes = file.read(4)
                dimensions.append(int.from_bytes(dimensionBytes, byteorder = 'big'))
            #print(dimensions)
            samplesCount = dimensions[0]
            if dimensionsAmount > 1:
                for i in range(1, dimensionsAmount):
                    sampleSize *= dimensions[i]
            #print(sampleSize)
            samples = file.read(samplesCount * sampleSize)
            return (dimensions, samples)
        else:
            print(f"Wrong MNIST file format {filename}")

def vectorized(i, n):
    vector = [0] * n
    vector[i] = 1
    return vector

#Check current network, by both training and test data. Used in mini batch SGD.
def evalMNIST():
    if globalValNetworkTrainExamplesQuantity > 0:
        correctTests = netLib.testNetworkByEvalData(
                globalValNetworkForEvaluationTest.contents,
                globalValNetworkTrainInputs,
                globalValNetworkTrainOutputs,
                globalValNetworkTrainExamplesQuantity
        )
        evalResults = correctTests / globalValNetworkTrainExamplesQuantity
        evaluationCorrectResultsOnTraining.append(evalResults)

    correctTests = netLib.testNetworkByEvalData(
            globalValNetworkForEvaluationTest.contents,
            globalValNetworkTestInputs,
            globalValNetworkTestOutputs,
            globalValNetworkTestExamplesQuantity
    )
    evalResults = correctTests / globalValNetworkTestExamplesQuantity
    evaluationCorrectResults.append(evalResults)
    return evalResults

#Create network with specific parameters and test it on MNIST base several times, draw graph and save results in json for future analysis.
def testMNIST(hiddenLayers, neuronsPerHiddenLayer, batchSize, maxEpochs, aftHidden, aftOutput, cft, startingLearningRate, fileName, l1RP=0, l2RP=0, dWD=0, nIEL=0, lRDL=0, beta1=0, beta2=0, eps=0, useAdamBiasCorrection=False, threading=True):
    evaluationCorrectResultsInAllTests = []
    evaluationCorrectResultsOnTrainingInAllTests = []

    # Prepare reusable image to draw results into.
    plt.figure(figsize=(10, 5 * testAmount), num=1, clear=True)

    for testIndex in range(testAmount):
        global evaluationCorrectResults
        evaluationCorrectResults = []

        global evaluationCorrectResultsOnTraining
        evaluationCorrectResultsOnTraining = []

        net = netLib.createNetwork(784, 10, hiddenLayers, neuronsPerHiddenLayer, batchSize, aftHidden, aftOutput, cft, ctypes.c_double(startingLearningRate), ctypes.c_double(beta1), ctypes.c_double(beta2), ctypes.c_double(eps), None, None)

        net.contents.l1RegularizationParameter = l1RP
        net.contents.l2RegularizationParameter = l2RP
        net.contents.decoupledWeightDecay = dWD
        net.contents.noImprovementsEpochsLimit = nIEL
        net.contents.learningRateDecreaserLimit = lRDL
        net.contents.biasCorrectionInAdamOptimizer = ctypes.c_bool(useAdamBiasCorrection)

        global globalValNetworkForEvaluationTest
        globalValNetworkForEvaluationTest = net
    
        if threading:
            netLib.startThreading()
        netLib.trainByMiniBatchStochasticGradientDescent(net.contents, trainInputs, trainResults, numberOfTrainSamples, 784, 10, maxEpochs, batchSize, evalFuncPointer)
        if threading:
            netLib.stopThreading()
        
        #netLib.printNetworkInFile(net.contents, fileName.encode('utf-8'), True)

        netLib.destroyNetwork(ctypes.byref(net))
        
        #print(evaluationCorrectResults)

        evaluationCorrectResultsInAllTests.append(evaluationCorrectResults)
        evaluationCorrectResultsOnTrainingInAllTests.append(evaluationCorrectResultsOnTraining)

        epochs = list(range(len(evaluationCorrectResults)))
    
        plt.subplot(testAmount, 1, testIndex + 1)
        lineTest, = plt.plot(epochs, evaluationCorrectResults, marker = 'o')
        testColor = lineTest.get_color()
        if evaluationCorrectResultsOnTraining:
            lineTrain, = plt.plot(epochs, evaluationCorrectResultsOnTraining, marker = '*')
            trainColor = lineTrain.get_color()
        plt.title(f'{testIndex}')
        plt.xlabel('Epochs')
        plt.ylabel('Correct MNIST')

        r, g, b, a = mcolors.to_rgba(testColor)
        testColor = (r / 2, g / 2, b / 2)
        if evaluationCorrectResultsOnTraining:
            r, g, b, a = mcolors.to_rgba(trainColor)
            trainColor = (r / 2, g / 2, b / 2)
        for i, res in enumerate(evaluationCorrectResults):
            plt.annotate(f'{i} {res:.4f}',
                         (epochs[i], evaluationCorrectResults[i]),
                         fontsize=3.5,
                         textcoords='offset points',
                         xytext=(0, 5),
                         ha='center',
                         color=testColor)

        if evaluationCorrectResultsOnTraining:
            for i, res in enumerate(evaluationCorrectResultsOnTraining):
                plt.annotate(f'{i} {res:.4f}',
                             (epochs[i], evaluationCorrectResultsOnTraining[i]),
                             fontsize=3.5,
                             textcoords='offset points',
                             xytext=(0, 5),
                             ha='center',
                             color=trainColor)


    resultsInJSON = {
            'hiddenLayers' : hiddenLayers,
            'neuronsPerHiddenLayer' : neuronsPerHiddenLayer,
            'batchSize' : batchSize,
            'maxEpochs' : maxEpochs,
            'aftHidden' : aftHidden,
            'aftOutput' : aftOutput,
            'cft' : cft,
            'startingLearningRate' : startingLearningRate,
            'l1RP' : l1RP,
            'l2RP' : l2RP,
            'dWD' : dWD,
            'nIEL' : nIEL,
            'lRDL' : lRDL,
            'beta1' : beta1,
            'beta2' : beta2,
            'eps' : eps,
            'useAdamBiasCorrection' : useAdamBiasCorrection,
            'threading' : threading,
            'evaluationResultsTest' : evaluationCorrectResultsInAllTests,
            'evaluationResultsTraining' : evaluationCorrectResultsOnTrainingInAllTests
    }
    testResultsDictionary.append(resultsInJSON)
    with open(testResultsFileName, 'w') as file:
        json.dump(testResultsDictionary, file, indent = 4)

    plt.suptitle(f'''hiddenLayers={hiddenLayers}, neuronsPerHiddenLayer={neuronsPerHiddenLayer}, batchSize={batchSize}, maxEpochs={maxEpochs},\naftHidden={aftHidden}, \
aftOutput={aftOutput}, cft={cft}, startingLearningRate={startingLearningRate}, l1RP={l1RP}, l2RP={l2RP}, dWD={dWD},\nnIEL={nIEL}, lRDL={lRDL}, \
beta1={beta1}, beta2={beta2}, eps={eps}, useAdamBiasCorrection={useAdamBiasCorrection}, treading={threading}''')

    os.makedirs('MNIST_ResultsImages', exist_ok = True)

    filePath = 'MNIST_ResultsImages/' + fileName + '.png'
    plt.savefig(filePath, dpi=300)

#Primitive experiments with autoencoders.
def testMNISTwithAutoencoder(hiddenLayers, neuronsPerHiddenLayer, batchSize, maxEpochs, aftHidden, aftOutput, cft, startingLearningRate, fileName, l1RP=0, l2RP=0, dWD=0, nIEL=0, lRDL=0, beta1=0, beta2=0, eps=0, useAdamBiasCorrection=False, threading=True):
    autoencoderMaxEpochs = 10
    autoencoderLearningRate = 0.01
    netAutoencoder = netLib.createNetwork(784, 784, 1, 50, batchSize, cActivationFunctionType.ReLU, cActivationFunctionType.linear, cCostFunctionType.square, ctypes.c_double(autoencoderLearningRate), ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0), None, None)

    netAutoencoder.contents.l1RegularizationParameter = 0.0005

    netLib.trainByMiniBatchStochasticGradientDescent(netAutoencoder.contents, trainInputs, trainInputs, numberOfTrainSamples, 784, 784, autoencoderMaxEpochs, batchSize, None)

    originalTrainInputsAddress = ctypes.addressof(trainInputs)
    originalTestInputsAddress = ctypes.addressof(testInputs)
    doubleSize = ctypes.sizeof(ctypes.c_double)

    preprocessedTrainInputs = arrTypeSamples()
    preprocessedTestInputs = arrTypeTestSamples()

    resultFirstIndex = netAutoencoder.contents.lastLayerFirstIndex
    for i in range(numberOfTrainSamples):
        currentTrainAddress = originalTrainInputsAddress + doubleSize * i * 784
        voidPointer = ctypes.c_void_p(currentTrainAddress)
        doublePointer = ctypes.cast(voidPointer, ctypes.POINTER(ctypes.c_double))
        netLib.calculate(netAutoencoder.contents, doublePointer, 0)

        for k in range(784):
            preprocessedTrainInputs[i * 784 + k] = netAutoencoder.contents.resData[resultFirstIndex + k]

    for i in range(numberOfTestSamples):
        currentTestAddress = originalTestInputsAddress + doubleSize * i * 784
        voidPointer = ctypes.c_void_p(currentTestAddress)
        doublePointer = ctypes.cast(voidPointer, ctypes.POINTER(ctypes.c_double))
        netLib.calculate(netAutoencoder.contents, doublePointer, 0)

        for k in range(784):
            preprocessedTestInputs[i * 784 + k] = netAutoencoder.contents.resData[resultFirstIndex + k]


    netLib.destroyNetwork(ctypes.byref(netAutoencoder))

    #Draw image from MNIST data.

    imagesCount = 10
    for imageIndex in range(imagesCount):
        startIndex = imageIndex * 784
        imageArray = [trainInputs[startIndex + i * 28:startIndex + (i + 1) * 28] for i in range(28)]
        #print('original')
        #for colors in imageArray:
        #    print(colors)
        plt.subplot(imagesCount, 2, imageIndex * 2 + 1)
        plt.imshow(imageArray, cmap='gray')
        imageArray = [preprocessedTrainInputs[startIndex + i * 28:startIndex + (i + 1) * 28] for i in range(28)]
        #print('processed')
        #for colors in imageArray:
        #    print(colors)

        plt.subplot(imagesCount, 2, imageIndex * 2 + 2)
        plt.imshow(imageArray, cmap='gray')

    plt.show()

    global globalValNetworkTestInputs
    globalValNetworkTestInputs = preprocessedTestInputs

    net = netLib.createNetwork(784, 10, hiddenLayers, neuronsPerHiddenLayer, batchSize, aftHidden, aftOutput, cft, ctypes.c_double(startingLearningRate), ctypes.c_double(beta1), ctypes.c_double(beta2), ctypes.c_double(eps), None, None)

    net.contents.l1RegularizationParameter = l1RP
    net.contents.l2RegularizationParameter = l2RP
    net.contents.decoupledWeightDecay = dWD
    net.contents.noImprovementsEpochsLimit = nIEL
    net.contents.learningRateDecreaserLimit = lRDL
    net.contents.biasCorrectionInAdamOptimizer = ctypes.c_bool(useAdamBiasCorrection)

    global globalValNetworkForEvaluationTest
    globalValNetworkForEvaluationTest = net
    
    if threading:
        netLib.startThreading()
    netLib.trainByMiniBatchStochasticGradientDescent(net.contents, preprocessedTrainInputs, trainResults, numberOfTrainSamples, 784, 10, maxEpochs, batchSize, evalFuncPointer)
    if threading:
        netLib.stopThreading()
    
    netLib.destroyNetwork(ctypes.byref(net))

#Process saved tests data, to choose best network configuration.
def processJson(condition = lambda x: True, fileName = 'MNIST, results of json processing'):
    with open(testResultsFileName, 'r') as file:
        jsonData = json.load(file)

        filtered = list(filter(condition, jsonData))

        for test in filtered:
            maxResults = []
            indexesOfMax = []
            meansOfLast5Epochs = []
            stdsOfLast5Epochs = []
            lastResults = []

            resultsDuringTests = test['evaluationResultsTest']
            for result in resultsDuringTests:
                last = result[-1]
                last5 = result[-5:]
                meanLast5 = np.mean(last5)
                stdLast5 = np.std(last5)
                indexOfMax = np.argmax(result)
                maxRes = result[indexOfMax]

                maxResults.append(maxRes)
                indexesOfMax.append(indexOfMax)
                meansOfLast5Epochs.append(meanLast5)
                stdsOfLast5Epochs.append(stdLast5)
                lastResults.append(last)

            averageLastResult = np.mean(lastResults)
            averageMeanOfLast5Epochs = np.mean(meansOfLast5Epochs)
            averageStdOfLast5Epochs = np.mean(stdsOfLast5Epochs)
            averageMax = np.mean(maxResults)
            averageEpochOfMax = np.mean(indexesOfMax)
            absoluteMax = max(maxResults)

            test['absoluteMax'] = absoluteMax
            test['averageLastResult'] = averageLastResult
            test['averageMeanOfLast5Epochs'] = averageMeanOfLast5Epochs
            test['averageStdOfLast5Epochs'] = averageStdOfLast5Epochs
            test['averageMax'] = averageMax
            test['averageEpochOfMax'] = averageEpochOfMax

        #Show best results.
        absoluteMax = 0
        averageMax = 0
        averageEpochOfMaxMax = 0
        averageEpochOfMaxMin = 1000000000
        averageMeanOfLast5Epochs = 0
        averageStdOfLast5Epochs = 100#This one is better minimized.
        averageLastResult = 0

        absoluteMaxTest = 0
        averageMaxTest = 0
        averageEpochOfMaxMaxTest = 0
        averageEpochOfMaxMinTest = 0
        averageMeanOfLast5EpochsTest = 0
        averageStdOfLast5EpochsTest = 0
        averageLastResultTest = 0

        for index, test in enumerate(jsonData):
            if test not in filtered: continue
            if test['absoluteMax'] > absoluteMax:
                absoluteMax = test['absoluteMax']
                absoluteMaxTest = index
            if test['averageLastResult'] > averageLastResult:
                averageLastResult = test['averageLastResult']
                averageLastResultTest = index
            if test['averageMeanOfLast5Epochs'] > averageMeanOfLast5Epochs:
                averageMeanOfLast5Epochs = test['averageMeanOfLast5Epochs']
                averageMeanOfLast5EpochsTest = index
            if test['averageStdOfLast5Epochs'] < averageStdOfLast5Epochs:
                averageStdOfLast5Epochs = test['averageStdOfLast5Epochs']
                averageStdOfLast5EpochsTest = index
            if test['averageMax'] > averageMax:
                averageMax = test['averageMax']
                averageMaxTest = index
            if test['averageEpochOfMax'] > averageEpochOfMaxMax:
                averageEpochOfMaxMax = test['averageEpochOfMax']
                averageEpochOfMaxMaxTest = index
            if test['averageEpochOfMax'] < averageEpochOfMaxMin:
                averageEpochOfMaxMin = test['averageEpochOfMax']
                averageEpochOfMaxMinTest = index

        parametersToShow = 7
        plt.figure(figsize=(10, 5 * parametersToShow), num=1, clear=True)
        plt.subplots_adjust(hspace = 0.5)
        drawJSONResults(jsonData[absoluteMaxTest], f'Highest absoluteMax: {absoluteMax}.', absoluteMaxTest, 0, parametersToShow)
        drawJSONResults(jsonData[averageLastResultTest], f'Highest averageLastResult: {averageLastResult}.', averageLastResultTest, 1, parametersToShow)
        drawJSONResults(jsonData[averageMeanOfLast5EpochsTest], f'Highest averageMeanOfLast5Epochs {averageMeanOfLast5Epochs}.', averageMeanOfLast5EpochsTest, 2, parametersToShow)
        drawJSONResults(jsonData[averageStdOfLast5EpochsTest], f'Lowest averageStdOfLast5Epochs {averageStdOfLast5Epochs}.', averageStdOfLast5EpochsTest, 3, parametersToShow)
        drawJSONResults(jsonData[averageMaxTest], f'Highest averageMax {averageMax}.', averageMaxTest, 4, parametersToShow)
        drawJSONResults(jsonData[averageEpochOfMaxMaxTest], f'Highest averageEpochOfMax {averageEpochOfMaxMax}.', averageEpochOfMaxMaxTest, 5, parametersToShow)
        drawJSONResults(jsonData[averageEpochOfMaxMinTest], f'Lowest averageEpochOfMax {averageEpochOfMaxMin}.', averageEpochOfMaxMinTest, 6, parametersToShow)
        plt.suptitle('Average results between tests for specific settings.') 
        plt.savefig(fileName, dpi=300)

#Draw saved experiment result.
def drawJSONResults(testJSON, paramName, testIndex, paramIndex, paramsAmount):
    plt.subplot(paramsAmount, 1, paramIndex + 1)

    averageResults = np.mean(testJSON['evaluationResultsTest'], axis=0)
    epochs = list(range(len(averageResults)))
    plt.plot(epochs, averageResults, marker = 'o')

    plt.xlabel('Epochs')
    plt.ylabel('Correct MNIST')

    for i, res in enumerate(averageResults):
        plt.annotate(f'{i} {res:.4f}',
                     (epochs[i], averageResults[i]),
                     fontsize=3.5,
                     textcoords='offset points',
                     xytext=(0, 5),
                     ha='center')

    plt.title(f'''{paramName} testIndex: {testIndex}\nhiddenLayers={testJSON['hiddenLayers']}, neuronsPerHiddenLayer={testJSON['neuronsPerHiddenLayer']}, \
batchSize={testJSON['batchSize']}, maxEpochs={testJSON['maxEpochs']},\naftHidden={testJSON['aftHidden']}, aftOutput={testJSON['aftOutput']}, \
cft={testJSON['cft']}, startingLearningRate={testJSON['startingLearningRate']}, l1RP={testJSON['l1RP']}, l2RP={testJSON['l2RP']}, \
dWD={testJSON['dWD']},\nnIEL={testJSON['nIEL']}, lRDL={testJSON['lRDL']}, beta1={testJSON['beta1']}, beta2={testJSON['beta2']}, \
eps={testJSON['eps']}, useAdamBiasCorrection={testJSON['useAdamBiasCorrection']}, treading={testJSON['threading']}''')


netLib.setupRandom()

evaluationCorrectResultsOnTraining = []
evaluationCorrectResults = []

# Simple XOR example.
#batchSize = 4
#net = netLib.createNetwork(2, 1, 1, 2, batchSize, cActivationFunctionType.sigmoid, cActivationFunctionType.sigmoid, cCostFunctionType.crossEntropy, ctypes.c_double(0.5), ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0), None, None)
#
#
#groupInputs = [1, 1,
#               1, 0,
#               0, 1,
#               0, 0]
#groupOutputs = [0,
#                1,
#                1,
#                0]
#arrType = ctypes.c_int * len(groupInputs)
#cGroupInputs = arrType(*groupInputs)
#
#arrType = ctypes.c_int * len(groupOutputs)
#cGroupOutputs = arrType(*groupOutputs)
#
#netLib.trainByMiniBatchStochasticGradientDescent(net.contents, cGroupInputs, cGroupOutputs, 4, 2, 1, 300, 2, None)
#exit()

trainImages = loadMNIST("train-images.idx3-ubyte")
trainLabels = loadMNIST("train-labels.idx1-ubyte")
testImages = loadMNIST("t10k-images.idx3-ubyte")
testLabels = loadMNIST("t10k-labels.idx1-ubyte")

numberOfTrainSamples = trainImages[0][0]
numberOfTestSamples = testImages[0][0]

# Prepare bare data for analysis.
resolution = trainImages[0][1] * trainImages[0][2]
normalizedTrainImages = [i/255.0 for i in trainImages[1]]
trainLabelsVectorized = [x for i in range(numberOfTrainSamples) for x in vectorized(trainLabels[1][i], 10)]
normalizedTestImages = [i/255.0 for i in testImages[1]]
testLabelsVectorized = [x for i in range(numberOfTestSamples) for x in vectorized(testLabels[1][i], 10)]

# Change train and test data to types used in network.
arrTypeSamples = ctypes.c_double * (784 * numberOfTrainSamples)
trainInputs = arrTypeSamples(*normalizedTrainImages)

arrTypeResults = ctypes.c_double * (10 * numberOfTrainSamples)
trainResults = arrTypeResults(*trainLabelsVectorized)

arrTypeTestSamples = ctypes.c_double * (784 * numberOfTestSamples)
testInputs = arrTypeTestSamples(*normalizedTestImages)

arrTypeTestResults = ctypes.c_double * (10 * numberOfTestSamples)
testResults = arrTypeTestResults(*testLabelsVectorized)

print("Loaded MNIST")

# Turned out there is bug, that existed more than 15 years - only simple types can be returned in callback functions, thus you can not just return struct. Decided to change C interface.
cFuncPointerType = ctypes.CFUNCTYPE(ctypes.c_double)#ctypes.CFUNCTYPE(cNetworkEvalResults)
evalFuncPointer = cFuncPointerType(evalMNIST)

# Draw image from MNIST data.
#imageIndex = 2
#startIndex = imageIndex * 784
#imageArray = [trainInputs[startIndex + i * 28:startIndex + (i + 1) * 28] for i in range(28)]
#plt.imshow(imageArray, cmap='gray')
#plt.show()
#print(trainResults[imageIndex * 10: imageIndex * 10 + 10])

globalValNetworkTestInputs = testInputs
globalValNetworkTestOutputs = testResults
globalValNetworkTestExamplesQuantity = numberOfTestSamples

globalValNetworkTrainInputs = trainInputs
globalValNetworkTrainOutputs = trainResults
globalValNetworkTrainExamplesQuantity = numberOfTrainSamples

testAmount = 10

testResultsFileName = 'MNIST_tests.json'
with open(testResultsFileName, 'r') as file:
    testResultsDictionary = json.load(file)

# Process saved results and finding the best network parameters by different metrics.
#processJson()
#processJson(condition = lambda test: test['eps'] > 0, fileName = 'MNIST, results of json processing, Adam optimizer')
#processJson(condition = lambda test: test['cft'] == 0, fileName = 'MNIST, results of json processing, square cost')
#exit()

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxEpochs = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.square,
#        startingLearningRate = 0.5,
#        fileName = 'MNIST_sigmoid_square_simplest',
#        l1RP = 0,
#        l2RP = 0,
#        dWD = 0,
#        nIEL = 0,
#        lRDL = 0,
#        beta1 = 0,
#        beta2 = 0,
#        eps = 0,
#        useAdamBiasCorrection = False,
#        threading = True
#)

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxEpochs = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.square,
#        startingLearningRate = 0.15,
#        fileName = 'MNIST_sigmoid_square_simplest_015'
#)
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxEpochs = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.square,
#        startingLearningRate = 0.5,
#        fileName = 'MNIST_sigmoid_square_05_nIEL5',
#        nIEL = 5,
#        lRDL = 0.0625
#)
#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxEpochs = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.crossEntropy,
#        startingLearningRate = 0.15,
#        fileName = 'MNIST_sigmoid_crossEntropy_simplest_015'
#)
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxEpochs = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.crossEntropy,
#        startingLearningRate = 0.15,
#        fileName = 'MNIST_sigmoid_crossEntropy_015_nIEL5_l2RP0001',
#        l2RP = 0.0001,
#        nIEL = 5,
#        lRDL = 0.0625
#)
#
testMNIST(
        hiddenLayers = 1,
        neuronsPerHiddenLayer = 30,
        batchSize = 10,
        maxEpochs = 30,
        aftHidden = cActivationFunctionType.sigmoid,
        aftOutput = cActivationFunctionType.softmax,
        cft = cCostFunctionType.logLikehood,
        startingLearningRate = 1.0,
        fileName = 'MNIST_softmax_logLikehood_1_nIEL1_l1RP00005_l2RP00001',
        l1RP = 0.00005,
        l2RP = 0.00001,
        nIEL = 1,
        lRDL = 0.015625
)

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxEpochs = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.softmax,
#        cft = cCostFunctionType.logLikehood,
#        startingLearningRate = 0.15,
#        fileName = 'MNIST_softmax_logLikehood_015_nIEL5_l2RP0001_beta1_05',
#        l2RP = 0.0001,
#        nIEL = 5,
#        lRDL = 0.0625,
#        beta1 = 0.5
#)

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxEpochs = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.square,
#        startingLearningRate = 0.001,
#        fileName = 'MNIST_sigmoid_square_Adam',
#        beta1 = 0.9,
#        beta2 = 0.999,
#        eps = 0.00000001,
#        useAdamBiasCorrection = True)

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxEpochs = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.crossEntropy,
#        startingLearningRate = 0.001,
#        fileName = 'MNIST_sigmoid_crossEntropy_Adam',
#        beta1 = 0.9,
#        beta2 = 0.999,
#        eps = 0.00000001,
#        useAdamBiasCorrection = True)
#

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxEpochs = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.softmax,
#        cft = cCostFunctionType.logLikehood,
#        startingLearningRate = 0.001,
#        fileName = 'MNIST_softmax_logLikehood_Adam_l1RP_00005',
#        l1RP = 0.00005,
##        l2RP = 0.00007,
##        dWD = 0.006,
#        beta1 = 0.9,
#        beta2 = 0.999,
#        eps = 0.00000001,
#        useAdamBiasCorrection = True)

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxEpochs = 30,
#        aftHidden = cActivationFunctionType.ReLU,
#        aftOutput = cActivationFunctionType.softmax,
#        cft = cCostFunctionType.logLikehood,
#        startingLearningRate = 0.001,
#        fileName = 'MNIST_ReLU_softmax_logLikehood_Adam_l1RP_00005',
#        l1RP = 0.00005,
##        l2RP = 0.00007,
##        dWD = 0.006,
#        beta1 = 0.9,
#        beta2 = 0.999,
#        eps = 0.00000001,
#        useAdamBiasCorrection = True)


