#include <stdio.h>
#include <stdbool.h>

// Just call srandom() to ensure unique initial weights for networks between application sessions. If for some reason you need the same network created repeatedly during different program launches - dont call it.
void setupRandom();

// Use extensive logging during training process, file nn_logs.txt will be created, if inConsole is false. For heavy debug purposes only.
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

// Originally, much of the data was inside neurons themselves, but during optimizations it was moved in arrays like resData in NeuralNetwork struct below. Now its purpose is faster access to precalculated data, that often used during training.
struct Neuron {
	int layer;
	int index;
	int weightsCount;
	enum ActivationFunctionType aft;
};

// Network itself.
struct NeuralNetwork {
	struct Neuron *net;
	int neuronsCount;// Total neurons count.
	int inputLayerNeuronsCount;
	int outputLayerNeuronsCount;
	int hiddenLayersCount;
	int neuronsPerHiddenLayer;
	enum CostFunctionType cft;
	int trainBlockSize;// Batch size during training. Must be known during network creation to allocate enough memory for training data.
	double baseTrainCoeff;// Initial multiplicator of gradient during training (learning rate). Can be reduced if no improvement for too long - depend on settings below.
	// Must be set manually after network creation, if you want to use these optimizations.
	double l2RegularizationParameter;
	int trainSamplesTotalAmount;// Used together with previous one for L2 regularization.
	int noImprovementsEpochsLimit;// If no improvement on test data during more than this amount of cycles - reduce learning rate by half.
	double trainCoeffCurrentDecreaser;// Multiplier for baseTrainCoeff, that is actually decreased then necessary.
	double trainCoeffDecreaserLimit;// Minimum for value above..
	double weightsMomentum;
	// Below are just internal data for training and optimizations.
	int lastLayerFirstIndex;// For optimizations, to not calculate every time it's needed.
	double *resData;// Results of neurons calculated for each block in batch, they go like this: allResults_block1, allResults_block2 ... allResults_block_last
	double *deltasData;// Deltas, calculated during backpropagation for each block in batch. Same order as results above.
	double *weights;
	double *bias;
	// Last velocities for weights and biases to use with momentum.
	double *weightsVelocities;
	double *biasVelocities;
};

// Network creation function. Most arguments explained in struct above. aftHidden - activation function type for hidden layers, aftOutput - for output one. loadedWeights and loadedBiases are used in loadNetwork function below, to set saved net info - only use if you REALLY know, what you are doing.
struct NeuralNetwork *createNetwork(int inputLayerNeuronsCount, int outputLayerNeuronsCount, int hiddenLayersCount, int neuronsPerHiddenLayer, int trainBlockSize, enum ActivationFunctionType aftHidden, enum ActivationFunctionType aftOutput, enum CostFunctionType cft, double baseTrainCoeff, double weightsMomentum, double *loadedWeights, double *loadedBiases);

void destroyNetwork(struct NeuralNetwork **nn);

void saveNetwork(struct NeuralNetwork nn, char *fileName);
// Load network from file, created by previous function. Last parameters are for training only and thefore are not saved.
struct NeuralNetwork *loadNetwork(char *fileName, int trainBlockSize, double baseTrainCoeff, double weightMomentum);

void calculate(struct NeuralNetwork nn, double *inputs, int resIndex);// Feedforward network. resIndex - index in the current batch, used to choose saving area in memory for results.

void printNetwork(struct NeuralNetwork nn);// Print network in console - only proper for very small nets.

void printNetworkInFile(struct NeuralNetwork nn, char *fileName);// Can be used for any network, unlike previous function.

double costFunction(struct NeuralNetwork nn, double *desiredOutputs, int samplesCount, FILE *logsOutput);// Calculate cost function for last calculated batch results. desiredOutputs - expected results for every batch in order (first all in first batch, then all for second etc). samplesCount - batchSize. logsOutput - file to log results and desirables for this calculation.

void trainByGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, double costFunctionToStop, int maxCycles);// Just train by one sample at the time, until do all of them in shuffled order, then repeat the cycle using new shuffle.

void trainByBatchGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, double costFunctionToStop, int maxCycles);// Use all samples in giant batch - for small sample size only.

void trainByMiniBatchStochasticGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, int maxCycles, int batchSize, double (*netCorrectness)());// Do SGD in mini batch way - train by using only samples from current batch, then repeat, until all samples processed, then next cycle shuffled again.

// Activate threading for network feedforward - very primitive implementation, thefore start and stop only immediately around network usage, to prevent wasting CPU resources.
void startThreading();
void stopThreading();

// For MNIST and other type-recognizing things, there one max value in output define result.
int testNetworkByEvalData(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, bool isTrain);
