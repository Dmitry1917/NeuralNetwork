import ctypes
from enum import IntEnum
import matplotlib.pyplot as plt

class cActivationFunctionType(IntEnum):
    sigmoid = 0
    tanhyp = 1
    ReLU = 2
    softmax = 3

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
            ('trainBlockSize', ctypes.c_int),
            ('baseTrainCoeff', ctypes.c_double),
            ('l2RegularizationParameter', ctypes.c_double),
            ('trainSamplesTotalAmount', ctypes.c_int),
            ('noImprovementsEpochsLimit', ctypes.c_int),
            ('trainCoeffCurrentDecreaser', ctypes.c_double),
            ('trainCoeffDecreaserLimit', ctypes.c_double),
            ('weightsMomentum', ctypes.c_double),
            ('lastLayerFirstIndex', ctypes.c_int),
            ('resData', ctypes.POINTER(ctypes.c_double)),
            ('deltasData', ctypes.POINTER(ctypes.c_double)),
            ('weights', ctypes.POINTER(ctypes.c_double)),
            ('bias', ctypes.POINTER(ctypes.c_double)),
            ('weightsVelocities', ctypes.POINTER(ctypes.c_double)),
            ('biasVelocities', ctypes.POINTER(ctypes.c_double))
    ]

def loadMNIST(fileName):
    file = open(fileName, "rb")
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

netLib = ctypes.CDLL('/home/dmitry/Documents/programs/C/nn/nn_net.so')

netLib.createNetwork.restype = ctypes.POINTER(cNeuralNetwork)
netLib.testNetworkByEvalData.restype = ctypes.c_int
netLib.testNetworkByEvalData.argtypes = [cNeuralNetwork, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_bool]

netLib.setupRandom()

evaluationCorrectResults = []

def evalMNIST():
    correctTests = netLib.testNetworkByEvalData(
            globalValNetworkForEvaluationTest.contents,
            globalValNetworkTestInputs,
            globalValNetworkTestOutputs,
            globalValNetworkTestExamplesQuantity,
            False
    )
    evaluationCorrectResults.append(correctTests)
    evalResults = correctTests / globalValNetworkTestExamplesQuantity
    return evalResults

# Simple XOR example.
#trainBlockSize = 4
#net = netLib.createNetwork(2, 1, 1, 2, trainBlockSize, cActivationFunctionType.sigmoid, cActivationFunctionType.sigmoid, cCostFunctionType.crossEntropy, ctypes.c_double(0.5), ctypes.c_double(0.0), None, None)
#
##print(net)
##print(net.contents.net[2].weightsCount)
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

# Prepare bare data for analisis.
resolution = trainImages[0][1] * trainImages[0][2]
normalizedTrainImages = [i/256.0 for i in trainImages[1]]
trainLabelsVectorized = [x for i in range(numberOfTrainSamples) for x in vectorized(trainLabels[1][i], 10)]
normalizedTestImages = [i/256.0 for i in testImages[1]]
testLabelsVectorized = [x for i in range(numberOfTestSamples) for x in vectorized(testLabels[1][i], 10)]

# Change train and test data to types used in network.
arrType = ctypes.c_double * (784 * numberOfTrainSamples)
trainInputs = arrType(*normalizedTrainImages)

arrType = ctypes.c_double * (10 * numberOfTrainSamples)
trainResults = arrType(*trainLabelsVectorized)

arrType = ctypes.c_double * (784 * numberOfTestSamples)
testInputs = arrType(*normalizedTestImages)

arrType = ctypes.c_double * (10 * numberOfTestSamples)
testResults = arrType(*testLabelsVectorized)

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

testAmount = 10

def testMNIST(hiddenLayers, neuronsPerHiddenLayer, trainBlockSize, maxTrainCycles, aftHidden, aftOutput, cft, startingLearningRate, weightMomentum, l2RP, nIEL, tCDL, fileName):
    # Prepare reusable image to draw results into.
    plt.figure(figsize=(10, 5 * testAmount), num=1, clear=True)

    for testIndex in range(testAmount):
        global evaluationCorrectResults
        evaluationCorrectResults = []

        net = netLib.createNetwork(784, 10, hiddenLayers, neuronsPerHiddenLayer, trainBlockSize, aftHidden, aftOutput, cft, ctypes.c_double(startingLearningRate), ctypes.c_double(weightMomentum), None, None)

        net.contents.l2RegularizationParameter = l2RP
        net.contents.trainSamplesTotalAmount = numberOfTrainSamples
        net.contents.noImprovementsEpochsLimit = nIEL
        net.contents.trainCoeffDecreaserLimit = tCDL

        global globalValNetworkForEvaluationTest
        globalValNetworkForEvaluationTest = net
    
        netLib.startThreading()
        netLib.trainByMiniBatchStochasticGradientDescent(net.contents, trainInputs, trainResults, numberOfTrainSamples, 784, 10, maxTrainCycles, trainBlockSize, evalFuncPointer)
        netLib.stopThreading()
        
        #netLib.printNetworkInFile(net.contents, fileName.encode('utf-8'))

        netLib.destroyNetwork(ctypes.byref(net))
        
        #print(evaluationCorrectResults)
        
        epochs = list(range(len(evaluationCorrectResults)))
    
        plt.subplot(testAmount, 1, testIndex + 1)
        plt.plot(epochs, evaluationCorrectResults, marker = 'o')
        plt.title(f'{testIndex}')
        plt.xlabel('Epochs')
        plt.ylabel('Correct MNIST')

        for i, res in enumerate(evaluationCorrectResults):
            plt.annotate(f'{i} {res}',
                         (epochs[i], evaluationCorrectResults[i]),
                         fontsize=4,
                         textcoords='offset points',
                         xytext=(0, 5),
                         ha='center')

    plt.suptitle(f'hiddenLayers={hiddenLayers}, neuronsPerHiddenLayer={neuronsPerHiddenLayer},\ntrainBlockSize={trainBlockSize}, maxTrainCycles={maxTrainCycles},\naftHidden={aftHidden}, aftOutput={aftOutput}, cft={cft}, startingLearningRate={startingLearningRate}, weightMomentum={weightMomentum},\nl2RP={l2RP}, nIEL={nIEL}, tCDL={tCDL}')
    plt.savefig(fileName, dpi=300)

#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.square,
#        startingLearningRate = 2.5,
#        weightMomentum = 0,
#        l2RP = 0,
#        nIEL = 0,
#        tCDL = 0,
#        fileName = 'MNIST_sigmoid_square_simplest')
#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.crossEntropy,
#        startingLearningRate = 0.5,
#        weightMomentum = 0,
#        l2RP = 0,
#        nIEL = 0,
#        tCDL = 0,
#        fileName = 'MNIST_sigmoid_crossEntropy_simplest')

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.square,
#        startingLearningRate = 2.5,
#        weightMomentum = 0,
#        l2RP = 0,
#        nIEL = 5,
#        tCDL = 0.0625,
#        fileName = 'MNIST_sigmoid_square_nIEL_5')
#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.crossEntropy,
#        startingLearningRate = 0.5,
#        weightMomentum = 0,
#        l2RP = 0,
#        nIEL = 5,
#        tCDL = 0.0625,
#        fileName = 'MNIST_sigmoid_crossEntropy_nIEL_5')

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.square,
#        startingLearningRate = 2.5,
#        weightMomentum = 0,
#        l2RP = 1.0,
#        nIEL = 5,
#        tCDL = 0.0625,
#        fileName = 'MNIST_sigmoid_square_nIEL_5_l2RP_1')
#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.crossEntropy,
#        startingLearningRate = 0.5,
#        weightMomentum = 0,
#        l2RP = 1.0,
#        nIEL = 5,
#        tCDL = 0.0625,
#        fileName = 'MNIST_sigmoid_crossEntropy_nIEL_5_l2RP_1')
#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.square,
#        startingLearningRate = 2.5,
#        weightMomentum = 0.1,
#        l2RP = 1.0,
#        nIEL = 5,
#        tCDL = 0.0625,
#        fileName = 'MNIST_sigmoid_square_nIEL_5_l2RP_1_wm_01')
#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.crossEntropy,
#        startingLearningRate = 0.5,
#        weightMomentum = 0.1,
#        l2RP = 1.0,
#        nIEL = 5,
#        tCDL = 0.0625,
#        fileName = 'MNIST_sigmoid_crossEntropy_nIEL_5_l2RP_1_wm_01')

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.softmax,
#        cft = cCostFunctionType.logLikehood,
#        startingLearningRate = 0.5,
#        weightMomentum = 0,
#        l2RP = 0,
#        nIEL = 0,
#        tCDL = 0,
#        fileName = 'MNIST_logLikehood_simplest')

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.softmax,
#        cft = cCostFunctionType.logLikehood,
#        startingLearningRate = 0.5,
#        weightMomentum = 0,
#        l2RP = 0,
#        nIEL = 5,
#        tCDL = 0.0625,
#        fileName = 'MNIST_logLikehood_nIEL_5')
#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.softmax,
#        cft = cCostFunctionType.logLikehood,
#        startingLearningRate = 0.5,
#        weightMomentum = 0,
#        l2RP = 1.0,
#        nIEL = 5,
#        tCDL = 0.0625,
#        fileName = 'MNIST_logLikehood_nIEL_5_l2RP_1')
#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        trainBlockSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.softmax,
#        cft = cCostFunctionType.logLikehood,
#        startingLearningRate = 0.5,
#        weightMomentum = 0.1,
#        l2RP = 1.0,
#        nIEL = 5,
#        tCDL = 0.0625,
#        fileName = 'MNIST_logLikehood_nIEL_5_l2RP_1_wm_01')
#

