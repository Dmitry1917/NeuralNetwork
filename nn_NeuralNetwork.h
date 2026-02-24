#include <stdio.h>
#include <stdbool.h>

void setupRandom();
void setupLogs(bool inConsole);
void closeLogs(); 

enum ActivationFunctionType {
	sigmoid,
	tanhyp,
	ReLU,
	softmax// Output layer only, with logLikehood cost function.
};

enum CostFunctionType {
	square,
	crossEntropy,// Only use with sigmoid output layer.
	logLikehood// Only use with softmax output layer. In fact it is actually crossEntropy for softmax activation function.
};

struct Neuron {
	int layer;
	int index;
	int weightsCount;
	enum ActivationFunctionType aft;
};

struct NeuralNetwork {
	struct Neuron *net;
	int neuronsCount;
	int inputLayerNeuronsCount;
	int outputLayerNeuronsCount;
	int hiddenLayersCount;
	int neuronsPerHiddenLayer;
	enum CostFunctionType cft;
	int trainBlockSize;
	double baseTrainCoeff;
	double l2RegularizationParameter;
	int trainSamplesTotalAmount;
	int noImprovementsEpochsLimit;
	double trainCoeffCurrentDecreaser;
	double trainCoeffDecreaserLimit;
	int lastLayerFirstIndex;
	// Results go like this: allResults_block1, allResults_block2 ... allResults_last
	double *resData;
	double *deltasData;
	double *weights;
	double *bias;
	double weightsMomentum;
	double *weightsVelocities;
};

struct NeuralNetwork *createNetwork(int inputLayerNeuronsCount, int outputLayerNeuronsCount, int hiddenLayersCount, int neuronsPerHiddenLayer, int trainBlockSize, enum ActivationFunctionType aftHidden, enum ActivationFunctionType aftOutput, enum CostFunctionType cft, double baseTrainCoeff, double weightsMomentum, double *loadedWeights, double *loadedBiases);

void destroyNetwork(struct NeuralNetwork **nn);

void saveNetwork(struct NeuralNetwork nn, char *fileName);

struct NeuralNetwork *loadNetwork(char *fileName, int trainBlockSize, double baseTrainCoeff, double weightMomentum);

void calculate(struct NeuralNetwork nn, double *inputs, int resIndex);

void printNetwork(struct NeuralNetwork nn);

void printNetworkInFile(struct NeuralNetwork nn);

double costFunction(struct NeuralNetwork nn, double *desiredOutputs, int samplesCount, FILE *logsOutput);

void trainByGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, double costFunctionToStop, int maxCycles);

void trainByBatchGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, double costFunctionToStop, int maxCycles);

void trainByMiniBatchStochasticGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, int maxCycles, int batchSize, double (*netCorrectness)());

// Threading.
void startThreading();
void stopThreading();

// For MNIST and other type-recognizing things, there one max value in output define result.
int testNetworkByEvalData(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, bool isTrain);
