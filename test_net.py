import ctypes
from enum import IntEnum
import matplotlib.pyplot as plt

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
netLib.testNetworkByEvalData.argtypes = [cNeuralNetwork, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]

netLib.setupRandom()

evaluationCorrectResults = []

def evalMNIST():
    correctTests = netLib.testNetworkByEvalData(
            globalValNetworkForEvaluationTest.contents,
            globalValNetworkTestInputs,
            globalValNetworkTestOutputs,
            globalValNetworkTestExamplesQuantity
    )
    evaluationCorrectResults.append(correctTests)
    evalResults = correctTests / globalValNetworkTestExamplesQuantity
    return evalResults

# Simple XOR example.
#batchSize = 4
#net = netLib.createNetwork(2, 1, 1, 2, batchSize, cActivationFunctionType.sigmoid, cActivationFunctionType.sigmoid, cCostFunctionType.crossEntropy, ctypes.c_double(0.5), ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0), None, None)
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

testAmount = 10

def testMNIST(hiddenLayers, neuronsPerHiddenLayer, batchSize, maxTrainCycles, aftHidden, aftOutput, cft, startingLearningRate, fileName, l1RP=0, l2RP=0, nIEL=0, lRDL=0, beta1=0, beta2=0, eps=0, useAdamBiasCorrection=False, threading=True):
    # Prepare reusable image to draw results into.
    plt.figure(figsize=(10, 5 * testAmount), num=1, clear=True)

    for testIndex in range(testAmount):
        global evaluationCorrectResults
        evaluationCorrectResults = []

        net = netLib.createNetwork(784, 10, hiddenLayers, neuronsPerHiddenLayer, batchSize, aftHidden, aftOutput, cft, ctypes.c_double(startingLearningRate), ctypes.c_double(beta1), ctypes.c_double(beta2), ctypes.c_double(eps), None, None)

        net.contents.l1RegularizationParameter = l1RP
        net.contents.l2RegularizationParameter = l2RP
        net.contents.noImprovementsEpochsLimit = nIEL
        net.contents.learningRateDecreaserLimit = lRDL
        net.contents.biasCorrectionInAdamOptimizer = ctypes.c_bool(useAdamBiasCorrection)

        global globalValNetworkForEvaluationTest
        globalValNetworkForEvaluationTest = net
    
        if threading:
            netLib.startThreading()
        netLib.trainByMiniBatchStochasticGradientDescent(net.contents, trainInputs, trainResults, numberOfTrainSamples, 784, 10, maxTrainCycles, batchSize, evalFuncPointer)
        if threading:
            netLib.stopThreading()
        
        #netLib.printNetworkInFile(net.contents, fileName.encode('utf-8'), True)

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

    plt.suptitle(f'hiddenLayers={hiddenLayers}, neuronsPerHiddenLayer={neuronsPerHiddenLayer},\nbatchSize={batchSize}, maxTrainCycles={maxTrainCycles},\naftHidden={aftHidden}, aftOutput={aftOutput}, cft={cft}, startingLearningRate={startingLearningRate}, l1RP={l1RP},\nl2RP={l2RP}, nIEL={nIEL}, lRDL={lRDL}, beta1={beta1}, beta2={beta2}, eps={eps}, useAdamBiasCorrection={useAdamBiasCorrection}, treading={threading}')
    plt.savefig(fileName, dpi=300)

def testMNISTwithAutoencoder(hiddenLayers, neuronsPerHiddenLayer, batchSize, maxTrainCycles, aftHidden, aftOutput, cft, startingLearningRate, fileName, l1RP=0, l2RP=0, nIEL=0, lRDL=0, beta1=0, beta2=0, eps=0, useAdamBiasCorrection=False, threading=True):
    # Prepare reusable image to draw results into.
    autoencoderMaxTrainCycles = 10
    autoencoderLearningRate = 0.01
    netAutoencoder = netLib.createNetwork(784, 784, 1, 50, batchSize, cActivationFunctionType.ReLU, cActivationFunctionType.linear, cCostFunctionType.square, ctypes.c_double(autoencoderLearningRate), ctypes.c_double(0.0), ctypes.c_double(0.0), ctypes.c_double(0.0), None, None)

    netAutoencoder.contents.l1RegularizationParameter = 0.0005

    netLib.trainByMiniBatchStochasticGradientDescent(netAutoencoder.contents, trainInputs, trainInputs, numberOfTrainSamples, 784, 784, autoencoderMaxTrainCycles, batchSize, None)

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
    net.contents.noImprovementsEpochsLimit = nIEL
    net.contents.learningRateDecreaserLimit = lRDL
    net.contents.biasCorrectionInAdamOptimizer = ctypes.c_bool(useAdamBiasCorrection)

    global globalValNetworkForEvaluationTest
    globalValNetworkForEvaluationTest = net
    
    if threading:
        netLib.startThreading()
    netLib.trainByMiniBatchStochasticGradientDescent(net.contents, preprocessedTrainInputs, trainResults, numberOfTrainSamples, 784, 10, maxTrainCycles, batchSize, evalFuncPointer)
    if threading:
        netLib.stopThreading()
    
    netLib.destroyNetwork(ctypes.byref(net))

testMNIST(
        hiddenLayers = 1,
        neuronsPerHiddenLayer = 30,
        batchSize = 10,
        maxTrainCycles = 30,
        aftHidden = cActivationFunctionType.sigmoid,
        aftOutput = cActivationFunctionType.sigmoid,
        cft = cCostFunctionType.square,
        startingLearningRate = 0.5,
        fileName = 'MNIST_sigmoid_square_simplest',
        l1RP = 0,
        l2RP = 0,
        nIEL = 0,
        lRDL = 0,
        beta1 = 0,
        beta2 = 0,
        eps = 0,
        useAdamBiasCorrection = False,
        threading = True
)

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxTrainCycles = 30,
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
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.square,
#        startingLearningRate = 0.15,
#        fileName = 'MNIST_sigmoid_square_015_nIEL5',
#        nIEL = 5,
#        lRDL = 0.0625
#)
#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxTrainCycles = 30,
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
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.crossEntropy,
#        startingLearningRate = 0.15,
#        fileName = 'MNIST_sigmoid_crossEntropy_015_nIEL5',
#        nIEL = 5,
#        lRDL = 0.0625
#)
#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.softmax,
#        cft = cCostFunctionType.logLikehood,
#        startingLearningRate = 0.15,
#        fileName = 'MNIST_softmax_logLikehood_simplest_015'
#)
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.softmax,
#        cft = cCostFunctionType.logLikehood,
#        startingLearningRate = 0.15,
#        fileName = 'MNIST_softmax_logLikehood_015_nIEL5',
#        nIEL = 5,
#        lRDL = 0.0625
#)

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.square,
#        startingLearningRate = 0.001,
#        fileName = 'MNIST_sigmoid_square_Adam',
#        l1RP = 0,
#        l2RP = 0,
#        nIEL = 0,
#        lRDL = 0,
#        beta1 = 0.9,
#        beta2 = 0.999,
#        eps = 0.00000001,
#        useAdamBiasCorrection = True,
#        threading = True)

#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.sigmoid,
#        cft = cCostFunctionType.crossEntropy,
#        startingLearningRate = 0.001,
#        fileName = 'MNIST_sigmoid_crossEntropy_Adam',
#        l1RP = 0,
#        l2RP = 0,
#        nIEL = 0,
#        lRDL = 0,
#        beta1 = 0.9,
#        beta2 = 0.999,
#        eps = 0.00000001,
#        useAdamBiasCorrection = True,
#        threading = True)
#
#testMNIST(
#        hiddenLayers = 1,
#        neuronsPerHiddenLayer = 30,
#        batchSize = 10,
#        maxTrainCycles = 30,
#        aftHidden = cActivationFunctionType.sigmoid,
#        aftOutput = cActivationFunctionType.softmax,
#        cft = cCostFunctionType.logLikehood,
#        startingLearningRate = 0.001,
#        fileName = 'MNIST_sigmoid_logLikehood_Adam',
#        l1RP = 0,
#        l2RP = 0,
#        nIEL = 0,
#        lRDL = 0,
#        beta1 = 0.9,
#        beta2 = 0.999,
#        eps = 0.00000001,
#        useAdamBiasCorrection = True,
#        threading = True)


