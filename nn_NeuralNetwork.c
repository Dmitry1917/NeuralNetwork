#include "nn_NeuralNetwork.h"
#include <assert.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <stdlib.h>

FILE *logsFile = NULL;
bool useThreading = false;

void setupRandom() {
	srandom(time(NULL));
}

// Logging.
void setupLogs(bool inConsole) {
	if(inConsole) logsFile = stdout;
	else logsFile = fopen("nn_logs.txt", "w");
}

void closeLogs() {
	if(logsFile != NULL) fclose(logsFile);
}

// Random Gauss distribution by Box-Muller transform. Function randomGauss below is one to use.
double randomUniformForBoxMullerMethod() {// Return value in (0;1], semiopen interval.
	return ((double)random() + 1.0) / (RAND_MAX + 1.0);
}

double randomGauss01() {
	double r1 = randomUniformForBoxMullerMethod();
	double r2 = randomUniformForBoxMullerMethod();
	return sqrt(-2.0 * log(r1)) * cos(2 * M_PI * r2);
}

double randomGauss(double mean, double deviation) {
	return randomGauss01() * deviation + mean;
}

// Used for training samples shuffling, to improve neural network learning.
void shuffle(int* array, int length) {
	if(length < 2) return;
	for(int i = 0; i < length; i++) {
		int index = rand() % length;
		int element = array[i];
		array[i] = array[index];
		array[index] = element;
	}
}

// Used to multiply neuron weights on its inputs.
double dotProduct(double *array1, double* array2, int count) {
	double res = 0;
	int i = 0;

	// Surprisingly good optimization - MNIST 10 cycles train and check time drop from 1:45 to 1:15 from this alone. Increasing hardcode more dont provide benefits.
	for(; i < count - 5; i += 5) {
		res += array1[i] * array2[i] +
			array1[i+1] * array2[i+1] +
			array1[i+2] * array2[i+2] +
			array1[i+3] * array2[i+3] +
			array1[i+4] * array2[i+4];
	}
	for(; i < count; i++) {
		res += array1[i] * array2[i];
	}

	return res;
}

// Helper functions for network saving and loading.
char* aftToString(enum ActivationFunctionType aft) {
	switch(aft) {
		case sigmoid:
			return "sigmoid";
		case tanhyp:
			return "tanhyp";
		case ReLU:
			return "ReLU";
		case softmax:
			return "softmax";
	}
}

enum ActivationFunctionType stringToAFT(char* str) {
	if(strcmp(str, "sigmoid") == 0) return sigmoid;
	if(strcmp(str, "tanhyp") == 0) return tanhyp;
	if(strcmp(str, "ReLU") == 0) return ReLU;
	if(strcmp(str, "softmax") == 0) return softmax;
	printf("\nUnknown activation function type %s\n, returning standart sigmoid as fallback", str);
	return sigmoid;
}

char* cftToString(enum CostFunctionType cft) {
	switch(cft) {
		case square:
			return "square";
		case crossEntropy:
			return "crossEntropy";
		case logLikehood:
			return "logLikehood";
	}
}

enum CostFunctionType stringToCFT(char* str) {
	if(strcmp(str, "square") == 0) return square;
	if(strcmp(str, "crossEntropy") == 0) return crossEntropy;
	if(strcmp(str, "logLikehood") == 0) return logLikehood;
	printf("\nUnknown cost function type %s\n, returning standart square as fallback", str);
	return square;
}

// Neuron activation function, based on its own inputs, thus softmax, that use all neurons in current layer to calculate each of its output, can't be done here.
double activation(double propagation, enum ActivationFunctionType aft) {
	switch(aft) {
		case sigmoid:
			return 1 / (1 + exp(-propagation));
		case tanhyp:
			double eInX = exp(propagation);
			double eInMinusX = exp(-propagation);
			return (eInX - eInMinusX) / (eInX + eInMinusX);
		case ReLU:
			return propagation < 0 ? 0 : propagation;
		case softmax:
			printf("\nNot supposed use softmax here.\n");
			exit(1);
	}
}

double derivativeOfLastActivation(double lastRes, enum ActivationFunctionType aft) {
	switch(aft) {
		case sigmoid:
			return lastRes * (1 - lastRes);
		case tanhyp:
			return 1 - lastRes * lastRes;
		case ReLU:
			return lastRes <= 0 ? 0 : 1;
		case softmax:
			printf("\nNot supposed use softmax here.\n");
			exit(1);
	}
}

// Used for creating new network or loading saved one, by additionally providing weights and biases.
struct NeuralNetwork *createNetwork(int inputLayerNeuronsCount, int outputLayerNeuronsCount, int hiddenLayersCount, int neuronsPerHiddenLayer, int trainBlockSize, enum ActivationFunctionType aftHidden, enum ActivationFunctionType aftOutput, enum CostFunctionType cft, double baseTrainCoeff, double weightsMomentum, double *loadedWeights, double *loadedBiases) {
	// Softmax can be only in the last layer.
	if(aftHidden == softmax) {
		printf("\nSoftmax can be only in the last layer.\n");
		exit(1);
	}

	int neuronsCount = inputLayerNeuronsCount + outputLayerNeuronsCount + hiddenLayersCount * neuronsPerHiddenLayer;
	struct NeuralNetwork *nn = malloc(sizeof(struct NeuralNetwork));

	nn->neuronsCount = neuronsCount;
	nn->inputLayerNeuronsCount = inputLayerNeuronsCount;
	nn->outputLayerNeuronsCount = outputLayerNeuronsCount;
	nn->hiddenLayersCount = hiddenLayersCount;
	nn->neuronsPerHiddenLayer = neuronsPerHiddenLayer;
	nn->trainBlockSize = trainBlockSize;
	nn->lastLayerFirstIndex = neuronsCount - outputLayerNeuronsCount;
	nn->cft = cft;
	nn->baseTrainCoeff = baseTrainCoeff;
	nn->l2RegularizationParameter = 0;
	nn->trainSamplesTotalAmount = 0;
	nn->noImprovementsEpochsLimit = 0;
	nn->trainCoeffCurrentDecreaser = 1;
	nn->trainCoeffDecreaserLimit = 1;

	nn->net = calloc(neuronsCount, sizeof(struct Neuron));
	nn->resData = calloc(trainBlockSize * neuronsCount, sizeof(double));
	nn->deltasData = calloc(trainBlockSize * neuronsCount, sizeof(double));
	// First hidden layer has number of weights equal to input layer neurons number, multiplied to number of neurons in layer itself, the same logic applied to other hidden layers and output layer.
	int weightsNumber;
	if(hiddenLayersCount > 0) {
		weightsNumber = inputLayerNeuronsCount * neuronsPerHiddenLayer + (hiddenLayersCount - 1) * neuronsPerHiddenLayer * neuronsPerHiddenLayer + neuronsPerHiddenLayer * outputLayerNeuronsCount;
	} else {
		weightsNumber = inputLayerNeuronsCount * outputLayerNeuronsCount;
	}
	if(loadedWeights == NULL) {
		nn->weights = malloc(weightsNumber * sizeof(double));
	} else {
		nn->weights = loadedWeights;
	}
	int biasNumber = neuronsCount - inputLayerNeuronsCount;
	if(loadedBiases == NULL) {
		nn->bias = malloc(biasNumber * sizeof(double));
	} else {
		nn->bias = loadedBiases;
	}

	nn->weightsMomentum = weightsMomentum;
	if(weightsMomentum > 0) {
		nn->weightsVelocities = calloc(weightsNumber, sizeof(double));
		nn->biasVelocities = calloc(biasNumber, sizeof(double));
	}
	// input layer
	for(int i = 0; i < inputLayerNeuronsCount; i++) {
		nn->net[i].layer = 0;
		nn->net[i].index = i;
		nn->net[i].weightsCount = 1;
		nn->net[i].aft = sigmoid;
	}

	int weightIndex = 0;
	// hidden layers
	for(int l = 0; l < hiddenLayersCount; l++) {
		for(int i = 0; i < neuronsPerHiddenLayer; i++) {
			int il = i + inputLayerNeuronsCount + l * neuronsPerHiddenLayer;
			nn->net[il].layer = 1 + l;
			nn->net[il].index = i;
			int previousLayerNeuronsCount;
			if(l == 0) {
				previousLayerNeuronsCount = inputLayerNeuronsCount;
			} else {
				previousLayerNeuronsCount = neuronsPerHiddenLayer;
			}
			
			nn->net[il].weightsCount = previousLayerNeuronsCount;
			nn->net[il].aft = aftHidden;

			if(loadedWeights == NULL) {
				double deviation = 1.0 / sqrt(previousLayerNeuronsCount);
				for(int k = 0; k < previousLayerNeuronsCount; k++) {
					double w = randomGauss(0, deviation);
					nn->weights[weightIndex] = w;
					weightIndex++;
				}
			}
			if(loadedBiases == NULL) {
				double bias = randomGauss(0, 1);
				nn->bias[il - inputLayerNeuronsCount] = bias;
			}
		}
	}

	// output layer
	for(int i = 0; i < outputLayerNeuronsCount; i++) {
		int il = i + nn->lastLayerFirstIndex;
		nn->net[il].layer = 1 + hiddenLayersCount;
		nn->net[il].index = i;
		int weightsCount = hiddenLayersCount > 0 ? neuronsPerHiddenLayer : inputLayerNeuronsCount;
		nn->net[il].weightsCount = weightsCount;
		nn->net[il].aft = aftOutput;

		if(loadedWeights == NULL) {
			double deviation = 1.0 / sqrt(weightsCount);
			for(int k = 0; k < weightsCount; k++) {
				double w = randomGauss(0, deviation);
				nn->weights[weightIndex] = w;
				weightIndex++;
			}
		}
		if(loadedBiases == NULL) {
			double bias = randomGauss(0, 1);
			nn->bias[il - inputLayerNeuronsCount] = bias;
		}
	}

	return nn;
}

void destroyNetwork(struct NeuralNetwork **nn) {
	free((**nn).net);
	free((**nn).resData);
	free((**nn).deltasData);
	free((**nn).weights);
	free((**nn).bias);
	if((**nn).weightsMomentum > 0) {
		free((**nn).weightsVelocities);
		free((**nn).biasVelocities);
	}
	free(*nn);
	*nn = NULL;
}

// Simple save and load, without optimizations or proper error handling.
void saveNetwork(struct NeuralNetwork nn, char *fileName) {
	FILE *file = fopen(fileName, "w");
	if(file == NULL) {
		printf("\nCould not open file %s to save network\n", fileName);
		return;
	}
	char buf[100];
	snprintf(buf, sizeof(buf), "inputLayerNeuronsCount %d\n", nn.inputLayerNeuronsCount);
	if(fputs(buf, file) == EOF) {
		printf("Could not write %s to %s\n", buf, fileName);
	}
	snprintf(buf, sizeof(buf), "outputLayerNeuronsCount %d\n", nn.outputLayerNeuronsCount);
	if(fputs(buf, file) == EOF) {
		printf("Could not write %s to %s\n", buf, fileName);
	}
	snprintf(buf, sizeof(buf), "hiddenLayersCount %d\n", nn.hiddenLayersCount);
	if(fputs(buf, file) == EOF) {
		printf("Could not write %s to %s\n", buf, fileName);
	}
	snprintf(buf, sizeof(buf), "neuronsPerHiddenLayer %d\n", nn.neuronsPerHiddenLayer);
	if(fputs(buf, file) == EOF) {
		printf("Could not write %s to %s\n", buf, fileName);
	}
	snprintf(buf, sizeof(buf), "aftHidden %s\n", aftToString(nn.net[nn.inputLayerNeuronsCount].aft));
	if(fputs(buf, file) == EOF) {
		printf("Could not write %s to %s\n", buf, fileName);
	}
	snprintf(buf, sizeof(buf), "aftOutput %s\n", aftToString(nn.net[nn.neuronsCount - 1].aft));
	if(fputs(buf, file) == EOF) {
		printf("Could not write %s to %s\n", buf, fileName);
	}
	snprintf(buf, sizeof(buf), "cft %s\n", cftToString(nn.cft));
	if(fputs(buf, file) == EOF) {
		printf("Could not write %s to %s\n", buf, fileName);
	}

	int weightsNumber;
	if(nn.hiddenLayersCount > 0) {
		weightsNumber = nn.inputLayerNeuronsCount * nn.neuronsPerHiddenLayer + (nn.hiddenLayersCount - 1) * nn.neuronsPerHiddenLayer * nn.neuronsPerHiddenLayer + nn.neuronsPerHiddenLayer * nn.outputLayerNeuronsCount;
	} else {
		weightsNumber = nn.inputLayerNeuronsCount * nn.outputLayerNeuronsCount;
	}
	int biasNumber = nn.neuronsCount - nn.inputLayerNeuronsCount;

	fputs("Weights:\n", file);

	for(int i = 0; i < weightsNumber; i++) {
		snprintf(buf, sizeof(buf), "%.17f\n", nn.weights[i]);
		if(fputs(buf, file) == EOF) {
			printf("Could not write weight %s to %s\n", buf, fileName);
		}
	}
	fputs("Biases:\n", file);
	for(int i = 0; i < biasNumber; i++) {
		snprintf(buf, sizeof(buf), "%.17f\n", nn.bias[i]);
		if(fputs(buf, file) == EOF) {
			printf("Could not write bias %s to %s\n", buf, fileName);
		}
	}
	fclose(file);
}

struct NeuralNetwork *loadNetwork(char *fileName, int trainBlockSize, double baseTrainCoeff, double weightMomentum) {
	FILE *file = fopen(fileName, "r");
	if(file == NULL) {
		printf("Could not open file %s for read.\n", fileName);
		return NULL;
	}
	char buf[100];

	int inputLayerNeuronsCount = 0;
	int outputLayerNeuronsCount = 0;
	int hiddenLayersCount = 0;
	int neuronsPerHiddenLayer = 0;
	enum ActivationFunctionType aftHidden;
	enum ActivationFunctionType aftOutput;
	enum CostFunctionType cft;

	if(fgets(buf, sizeof(buf), file) != NULL) {
		char *partOfSplit = strtok(buf, " ");
		partOfSplit = strtok(NULL, " ");
		inputLayerNeuronsCount = strtol(partOfSplit, NULL, 10);
		printf("inputLayerNeuronsCount is %d\n", inputLayerNeuronsCount);
	}
	if(fgets(buf, sizeof(buf), file) != NULL) {
		char *partOfSplit = strtok(buf, " ");
		partOfSplit = strtok(NULL, " ");
		outputLayerNeuronsCount = strtol(partOfSplit, NULL, 10);
		printf("outputLayerNeuronsCount is %d\n", outputLayerNeuronsCount);
	}
	if(fgets(buf, sizeof(buf), file) != NULL) {
		char *partOfSplit = strtok(buf, " ");
		partOfSplit = strtok(NULL, " ");
		hiddenLayersCount = strtol(partOfSplit, NULL, 10);
		printf("hiddenLayersCount is %d\n", hiddenLayersCount);
	}
	if(fgets(buf, sizeof(buf), file) != NULL) {
		char *partOfSplit = strtok(buf, " ");
		partOfSplit = strtok(NULL, " ");
		neuronsPerHiddenLayer = strtol(partOfSplit, NULL, 10);
		printf("neuronsPerHiddenLayer is %d\n", neuronsPerHiddenLayer);
	}
	if(fgets(buf, sizeof(buf), file) != NULL) {
		char *partOfSplit = strtok(buf, " ");
		partOfSplit = strtok(NULL, " ");
		partOfSplit[strcspn(partOfSplit, "\n")] = 0;// Get rid of eol at the end.
		aftHidden = stringToAFT(partOfSplit);
		printf("aftHidden is %s %d\n", partOfSplit, aftHidden);
	}
	if(fgets(buf, sizeof(buf), file) != NULL) {
		char *partOfSplit = strtok(buf, " ");
		partOfSplit = strtok(NULL, " ");
		partOfSplit[strcspn(partOfSplit, "\n")] = 0;// Get rid of eol at the end.
		aftOutput = stringToAFT(partOfSplit);
		printf("aftOutput is %s %d\n", partOfSplit, aftOutput);
	}
	if(fgets(buf, sizeof(buf), file) != NULL) {
		char *partOfSplit = strtok(buf, " ");
		partOfSplit = strtok(NULL, " ");
		partOfSplit[strcspn(partOfSplit, "\n")] = 0;// Get rid of eol at the end.
		cft = stringToCFT(partOfSplit);
		printf("cft is %s %d\n", partOfSplit, cft);
	}
	if(fgets(buf, sizeof(buf), file) != NULL) {
		if(strcmp(buf, "Weights:\n") != 0) printf("Did not find weights label\n");
	}
	int weightsNumber;
	if(hiddenLayersCount > 0) {
		weightsNumber = inputLayerNeuronsCount * neuronsPerHiddenLayer + (hiddenLayersCount - 1) * neuronsPerHiddenLayer * neuronsPerHiddenLayer + neuronsPerHiddenLayer * outputLayerNeuronsCount;
	} else {
		weightsNumber = inputLayerNeuronsCount * outputLayerNeuronsCount;
	}
	double *weights = malloc(weightsNumber * sizeof(double));
	for(int i = 0; i < weightsNumber; i++) {
		if(fgets(buf, sizeof(buf), file) != NULL) {
			double weight = strtod(buf, NULL);
			weights[i] = weight;
			//printf("weight %d is %f\n", i, weight);
		}
	}

	if(fgets(buf, sizeof(buf), file) != NULL) {
		if(strcmp(buf, "Biases:\n") != 0) printf("Did not find biases label\n");
	}
	int neuronsCount = inputLayerNeuronsCount + outputLayerNeuronsCount + hiddenLayersCount * neuronsPerHiddenLayer;
	int biasesNumber = neuronsCount - inputLayerNeuronsCount;
	double *bias = malloc((biasesNumber) * sizeof(double));
	for(int i = 0; i < biasesNumber; i++) {
		if(fgets(buf, sizeof(buf), file) != NULL) {
			double biasI = strtod(buf, NULL);
			bias[i] = biasI;
			//printf("bias %d is %f\n", i, biasI);
		}
	}
	fclose(file);

	struct NeuralNetwork *nn = createNetwork(inputLayerNeuronsCount, outputLayerNeuronsCount, hiddenLayersCount, neuronsPerHiddenLayer, trainBlockSize, aftHidden, aftOutput, cft, baseTrainCoeff, weightMomentum, weights, bias);
	return nn;
}

// Separate network feedforward calculations on 4 threads.
struct ProcessActivationData {
	enum ActivationFunctionType aft;
	int weightsNumber;
	double *prevLayerResults;
	double *weights;
	double *bias;
	double *resData;
	int neuronsCount;
};

sem_t sem1, sem2, sem3;
pthread_t thread1, thread2, thread3;
struct ProcessActivationData *pad1, *pad2, *pad3;

void *processActivationQueue1(void *args) {
	int prevType;
	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &prevType);

	while(1) {
		//usleep(1);// For valgrind - otherwise it takes too long to check.
		if(pad1) {
			double *weights = pad1->weights;
			double *bias = pad1->bias;
			double *resData = pad1->resData;
			for(int i = 0; i < pad1->neuronsCount; i++) {
				double propagation = dotProduct(weights, pad1->prevLayerResults, pad1->weightsNumber);
				propagation += *bias;
				weights += pad1->weightsNumber;
				bias++;
				*resData = activation(propagation, pad1->aft);
				resData++;
			}
			pad1 = NULL;
			sem_post(&sem1);
		}
	}
}

void *processActivationQueue2(void *args) {
	int prevType;
	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &prevType);

	while(1) {
		//usleep(1);// For valgrind - otherwise it takes too long to check.
		if(pad2) {
			double *weights = pad2->weights;
			double *bias = pad2->bias;
			double *resData = pad2->resData;
			for(int i = 0; i < pad2->neuronsCount; i++) {
				double propagation = dotProduct(weights, pad2->prevLayerResults, pad2->weightsNumber);
				propagation += *bias;
				weights += pad2->weightsNumber;
				bias++;
				*resData = activation(propagation, pad2->aft);
				resData++;
			}
			pad2 = NULL;
			sem_post(&sem2);
		}
	}
}

void *processActivationQueue3(void *args) {
	int prevType;
	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &prevType);

	while(1) {
		//usleep(1);// For valgrind - otherwise it takes too long to check.
		if(pad3) {
			double *weights = pad3->weights;
			double *bias = pad3->bias;
			double *resData = pad3->resData;
			for(int i = 0; i < pad3->neuronsCount; i++) {
				double propagation = dotProduct(weights, pad3->prevLayerResults, pad3->weightsNumber);
				propagation += *bias;
				weights += pad3->weightsNumber;
				bias++;
				*resData = activation(propagation, pad3->aft);
				resData++;
			}
			pad3 = NULL;
			sem_post(&sem3);
		}
	}
}

void startThreading() {
	// Detached threads free memory after cancel.
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, 1);

	sem_init(&sem1, 0, 0);
	sem_init(&sem2, 0, 0);
	sem_init(&sem3, 0, 0);
	pthread_create(&thread1, &attr, processActivationQueue1, NULL);
	pthread_create(&thread2, &attr, processActivationQueue2, NULL);
	pthread_create(&thread3, &attr, processActivationQueue3, NULL);
	pthread_attr_destroy(&attr);

	useThreading = true;
}

void stopThreading() {
	useThreading = false;

	pthread_cancel(thread1);
	pthread_cancel(thread2);
	pthread_cancel(thread3);
	sem_destroy(&sem1);
	sem_destroy(&sem2);
	sem_destroy(&sem3);
}

// Feedforward data to network. resIndex parameter show index of sample in mini batch, to save results properly for future use during learning.
void calculate(struct NeuralNetwork nn, double *inputs, int resIndex) {
	/*if(resIndex < 0) {
		fprintf(stderr, "\nresult index < 0\n");
		exit(1);
	}*/
	int resIndexShift = nn.neuronsCount * resIndex;
	for(int i = 0; i < nn.inputLayerNeuronsCount; i++) {
		nn.resData[resIndexShift + i] = inputs[i];
	}

	// First hidden layer.
	double *weights = nn.weights;
	double *bias = nn.bias;
	double *prevLayerRes = nn.resData + resIndexShift;
	double *resData = prevLayerRes + nn.inputLayerNeuronsCount;

	struct ProcessActivationData data1, data2, data3;
	// 0 below means first hidden layer, and hiddenLayersCount included means outer layer.
	for(int layer = 0; layer <= nn.hiddenLayersCount; layer++) {
		int weightsNumber = layer == 0 ? nn.inputLayerNeuronsCount : nn.neuronsPerHiddenLayer;
		enum ActivationFunctionType aft = layer == nn.hiddenLayersCount ? nn.net[nn.neuronsCount - 1].aft : nn.net[nn.inputLayerNeuronsCount].aft;
		int neuronsCount = layer == nn.hiddenLayersCount ? nn.outputLayerNeuronsCount : nn.neuronsPerHiddenLayer;

		int previousNeurons = 0;
		bool useThreadingForThisLayer = useThreading && aft != softmax;

		if(useThreadingForThisLayer) {
			data1.weightsNumber = weightsNumber;
			data1.aft = aft;
			data1.prevLayerResults = prevLayerRes;
			data1.weights = weights;
			data1.bias = bias;
			data1.resData = resData;
			data1.neuronsCount = neuronsCount / 4;
		
			data2.weightsNumber = weightsNumber;
			data2.aft = aft;
			data2.prevLayerResults = prevLayerRes;
			data2.weights = weights + data1.neuronsCount * weightsNumber;
			data2.bias = bias + data1.neuronsCount;
			data2.resData = resData + data1.neuronsCount;
			data2.neuronsCount = data1.neuronsCount;
		
			data3.weightsNumber = weightsNumber;
			data3.aft = aft;
			data3.prevLayerResults = prevLayerRes;
			data3.weights = weights + 2 * data1.neuronsCount * weightsNumber;
			data3.bias = bias + 2 * data1.neuronsCount;
			data3.resData = resData + 2 * data1.neuronsCount;
			data3.neuronsCount = data1.neuronsCount;
		
			previousNeurons = 3 * data1.neuronsCount;
			weights += previousNeurons * weightsNumber;
			bias += previousNeurons;
			resData += previousNeurons;
		
			pad1 = &data1;
			pad2 = &data2;
			pad3 = &data3;
		}
		if(aft == softmax) {
			double *exponents = malloc(nn.outputLayerNeuronsCount * sizeof(double));
			double expSum = 0;
			for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
				double propagation = dotProduct(weights, prevLayerRes, nn.neuronsPerHiddenLayer);
				propagation += *bias;
				double exponent = exp(propagation);
				exponents[i] = exponent;
				expSum += exponent;
				weights += nn.neuronsPerHiddenLayer;
				bias++;
			}
			for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
				*resData = exponents[i] / expSum;
				resData++;
			}
			free(exponents);
		} else {
			for(int i = previousNeurons; i < neuronsCount; i++) {
				double propagation = dotProduct(weights, prevLayerRes, weightsNumber);
				propagation += *bias;
				weights += weightsNumber;
				bias++;
				*resData = activation(propagation, aft);
				resData++;
			}
		}

		if(useThreadingForThisLayer) {
			sem_wait(&sem1);
			sem_wait(&sem2);
			sem_wait(&sem3);
		}
		prevLayerRes += weightsNumber;
	}
}

// Use for small networks only and one result (if batches are used, only first result will be printed).
void printNetwork(struct NeuralNetwork nn) {
	//printf("\x1b[2B");
	printf("\n");
	printf("nn neuronsCount %d inputLayerNeuronsCount %d outputLayerNeuronsCount %d hiddenLayersCount %d neuronsPerHiddenLayer %d\n", nn.neuronsCount, nn.inputLayerNeuronsCount, nn.outputLayerNeuronsCount, nn.hiddenLayersCount, nn.neuronsPerHiddenLayer);

	printf("\n");
	int widthReserve = 7;
	int maxHeight = nn.inputLayerNeuronsCount > nn.outputLayerNeuronsCount ? nn.inputLayerNeuronsCount : nn.outputLayerNeuronsCount;
	maxHeight = maxHeight > nn.neuronsPerHiddenLayer ? maxHeight : nn.neuronsPerHiddenLayer;
	int lastLayer = -1;
	int lastEolXCoord = 0;
	double *weights = nn.weights;
	double *bias = nn.bias;
	for(int i = 0; i < nn.neuronsCount; i++) {
		struct Neuron n = nn.net[i];
		int layer = 0;
		int indexInLayer = i;
		if(i >= nn.inputLayerNeuronsCount) {
			if(i < nn.neuronsCount - nn.outputLayerNeuronsCount) {
				layer = (i - nn.inputLayerNeuronsCount) / nn.neuronsPerHiddenLayer + 1;
				indexInLayer = (i - nn.inputLayerNeuronsCount) % nn.neuronsPerHiddenLayer;
			}
			else {
				layer = nn.hiddenLayersCount + 1;
				indexInLayer = nn.outputLayerNeuronsCount - (nn.neuronsCount - i);
			}
		}

		int layerWidth = layer == 0 ? 9 : 7 * n.weightsCount + 16 + widthReserve;
		for(int k = 0; k < indexInLayer; k++) {
			printf("\n");
		}
		if(n.index == 0) {
			printf("\r");
		}

		//int xCoord = n.layer * (perLayer + betweenLayers);
		if(lastLayer < layer) {
			lastLayer = layer;
			lastEolXCoord += layerWidth;
		}
		int xCoord = lastEolXCoord - layerWidth; 

		// move cursor right
		printf("\x1b[%dC", xCoord);
		if(n.layer == 0) {
			printf("%.4f", nn.resData[i]);
		} else {
			for(int w = 0; w < n.weightsCount; w++) {
				printf("%.4f ", *weights);
				weights++;
			}
			printf("%.4f ", *bias);
			bias++;
			//printf("%d %d %d", i, xCoord, n.weightsCount);
			printf("%.4f %.3f", nn.resData[i], nn.deltasData[i]);

		}
		if(indexInLayer > 0) {
			// move cursor up specified number of lines
			printf("\x1b[%dA", indexInLayer);
		}
	}
	for(int k = 0; k < maxHeight; k++) {
		printf("\n");
	}
}

// Can be used for any network and print batch results too.
void printNetworkInExistingFILE(struct NeuralNetwork nn, FILE *file) {
	if(file == NULL) return;
	int maxLevel = nn.inputLayerNeuronsCount;
	if(nn.outputLayerNeuronsCount > maxLevel) maxLevel = nn.outputLayerNeuronsCount;
	if(nn.neuronsPerHiddenLayer > maxLevel) maxLevel = nn.neuronsPerHiddenLayer;

	int level = 0;
	int i = 0;
	while(level < maxLevel) {
		struct Neuron n = nn.net[i];
		if(n.layer == 0) {
			fprintf(file, "\nneuron ");
			for(int r = 0; r < nn.trainBlockSize; r++) {
				fprintf(file, "res%d: %.4f ", r, nn.resData[nn.neuronsCount * r + i]);
			}
		} else {
			fprintf(file, "neuron layer %d index %d weights: ", n.layer, n.index);
			int firstWeightIndex;
			if(n.layer == 1) {
				firstWeightIndex = nn.inputLayerNeuronsCount * n.index;
			} else {
				firstWeightIndex = nn.neuronsPerHiddenLayer * nn.inputLayerNeuronsCount + (n.layer - 2) * nn.neuronsPerHiddenLayer * nn.neuronsPerHiddenLayer + n.index * nn.neuronsPerHiddenLayer;
			}
			for(int w = 0; w < n.weightsCount; w++) {
				fprintf(file, "%.4f ", nn.weights[firstWeightIndex + w]);
			}
			fprintf(file, "bias: %.4f ", nn.bias[i - nn.inputLayerNeuronsCount]);
			//printf("%d %d %d", i, xCoord, n.weightsCount);
			for(int r = 0; r < nn.trainBlockSize; r++) {
				fprintf(file, "res%d: %.4f delta%d: %.3f ", r, nn.resData[nn.neuronsCount * r + i], r, nn.deltasData[nn.neuronsCount * r + i]);
			}
		}

		fprintf(file, "     ");

		if(i < nn.inputLayerNeuronsCount) {
			if(level < nn.neuronsPerHiddenLayer) {
				i += nn.inputLayerNeuronsCount;
			} else if(level < nn.outputLayerNeuronsCount) {
				i += nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * nn.hiddenLayersCount;
			} else if(i == nn.inputLayerNeuronsCount - 1) break;
			else i = nn.neuronsCount;// make sure that level increase in only one place
		}
		else if(i < nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * nn.hiddenLayersCount) {
			// if after this go beyound last layer, then it means it is shorter than hidden ones
			i += nn.neuronsPerHiddenLayer;
		}
		else if(level < nn.inputLayerNeuronsCount || level < nn.neuronsPerHiddenLayer) i += nn.outputLayerNeuronsCount;// next level afterwards
		else i = nn.neuronsCount;// make sure that level increase in only one place

		if(i >= nn.neuronsCount) {
			level++;
			fprintf(file, "\n");
			// find first layer with enough neurons
			if(level < nn.inputLayerNeuronsCount) i = level;
			else if(level < nn.neuronsPerHiddenLayer) i = level + nn.inputLayerNeuronsCount;
			else i = nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * nn.hiddenLayersCount + level;
		}
	}
}

void printNetworkInFile(struct NeuralNetwork nn, char *fileName) {
	FILE *file = fopen(fileName, "a");
	if(file == NULL) return;

	fprintf(file, "\n\nNext network.\n\n");
	printNetworkInExistingFILE(nn, file);
	fclose(file);
}

#define nanPreventionLimit 0.0001
double costFunction(struct NeuralNetwork nn, double *desiredOutputs, int samplesCount, FILE *logsOutput) {
	/*if(samplesCount < 1) {
		fprintf(stderr, "\ntrainSize < 1\n");
		exit(1);
	}*/
	double cost = 0;

	switch(nn.cft) {
		case square:
			for(int resIndex = 0; resIndex < samplesCount; resIndex++) {
				double sampleCost = 0;
				for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
					sampleCost += pow(nn.resData[nn.neuronsCount * resIndex + nn.lastLayerFirstIndex + i] - desiredOutputs[nn.outputLayerNeuronsCount * resIndex + i], 2);
				}
				cost += sampleCost * 0.5;
			}
			break;
		case crossEntropy:
			for(int resIndex = 0; resIndex < samplesCount; resIndex++) {
				double sampleCost = 0;
				for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
					// Intentionaly written very detailed for easier understanding. Performance here isn't noticably affected.
					double desired = desiredOutputs[nn.outputLayerNeuronsCount * resIndex + i];
					double real = nn.resData[nn.neuronsCount * resIndex + nn.lastLayerFirstIndex + i];
					double realSubtracted = 1 - real;
					// Without such check can sometimes get nan, because of float rounding leading to zero in logarithm.
					if(real < nanPreventionLimit) real = nanPreventionLimit;
					if(realSubtracted < nanPreventionLimit) realSubtracted = nanPreventionLimit;
					sampleCost += -(desired * log(real) + (1 - desired) * log(realSubtracted));
				}
				cost += sampleCost;
			}
			break;
		case logLikehood:
			// This can be used only with activation functions, that leads to final results resembling probability distribution and desired results being 0 and 1 only.
			for(int resIndex = 0; resIndex < samplesCount; resIndex++) {
				double sampleCost = 0;
				for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
					double desired = desiredOutputs[nn.outputLayerNeuronsCount * resIndex + i];
					// Prevent floating point comparison issues again.
					if(desired > 0.999) {
						double real = nn.resData[nn.neuronsCount * resIndex + nn.lastLayerFirstIndex + i];
						if(real < nanPreventionLimit) real = nanPreventionLimit;
						sampleCost = -log(real);
						break;
					}
					if(i == nn.outputLayerNeuronsCount - 1) {
						printf("\nNot supposed to come here - desired results must have 1 if correct.\n");
						exit(1);
					}
				}
				cost += sampleCost;
			}
			break;
	}

	// In all cases, there only part of samples is used, it will be later summed and divided by number of groups of samples, and that will made it like dividing all costs on all samples, like intended. To be completely fair - if total samples amount is not divisible on mini batch size, then there will be some error, but in all real cases (many samples, limited mini batch) it will be minor and inconsequential, because cost is used only for some control, not in calculations themselfs.
	cost /= samplesCount;

	// L2 regularization.
	if(nn.l2RegularizationParameter > 0) {
		int weightsCount;
		if(nn.hiddenLayersCount > 0) {
			weightsCount = nn.inputLayerNeuronsCount * nn.neuronsPerHiddenLayer + (nn.hiddenLayersCount - 1) * nn.neuronsPerHiddenLayer * nn.neuronsPerHiddenLayer + nn.neuronsPerHiddenLayer * nn.outputLayerNeuronsCount;
		} else {// perceptron
			weightsCount = nn.inputLayerNeuronsCount * nn.outputLayerNeuronsCount;
		}
		double weightsSquaresSum = 0;
		double *weights = nn.weights;
		for(int i = 0; i < weightsCount; i++) {
			weightsSquaresSum += *weights * *weights;
			weights++;
		}
		cost += nn.l2RegularizationParameter * weightsSquaresSum * 0.5 / nn.trainSamplesTotalAmount;
	}

	if(logsOutput != NULL) {
		/*fprintf(logsOutput, "\ninputs:");
		for(int block = 0; block < trainBlockSize; block++) {
			fprintf(logsOutput, "\n    inputs%d:", block);
			for(int i = 0; i < nn.inputLayerNeuronsCount; i++) {
				fprintf(logsOutput, " %f", nn.net[i].results[block]);
			}
		}*/
	
		fprintf(logsOutput, "\nresults:");
		for(int block = 0; block < samplesCount; block++) {
			fprintf(logsOutput, "\n    results%d:", block);
			for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
				fprintf(logsOutput, " %f", nn.resData[nn.neuronsCount * block + nn.lastLayerFirstIndex + i]);
			}
		}
	
		fprintf(logsOutput, "\nwe need:");//text like this to make desired results right under real ones
		for(int block = 0; block < samplesCount; block++) {
			fprintf(logsOutput, "\n    we need%d:", block);
			for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
				fprintf(logsOutput, " %f", desiredOutputs[nn.outputLayerNeuronsCount * block + i]);
			}
		}
		fprintf(logsOutput, "\ncost function %f\n", cost);
	}

	return cost;
}

void calculateDeltas(struct NeuralNetwork nn, double *results, int resIndex) {
	/*if(resIndex < 0) {
		fprintf(stderr, "\nresIndex < 0\n");
		exit(1);
	}*/
	// neurons
	int i = nn.neuronsCount - 1;
	int resIndexShift = nn.neuronsCount * resIndex;

	// Last layer process separately for a bit of performance. And later for cross entropy implementation.
	switch(nn.cft) {
		case square:
			for(; i >= nn.lastLayerFirstIndex; i--) {
				struct Neuron *n = &(nn.net[i]);
				double lastRes = nn.resData[resIndexShift + i]; 
		
				double dErrorBydSigma = lastRes - results[i - nn.lastLayerFirstIndex];
				double lastActivationDerivative = derivativeOfLastActivation(lastRes, n->aft);
				double delta = lastActivationDerivative * dErrorBydSigma;
				nn.deltasData[resIndexShift + i] = delta;
			}
			break;
		case crossEntropy:
		case logLikehood:
			for(; i >= nn.lastLayerFirstIndex; i--) {
				struct Neuron *n = &(nn.net[i]);
				double lastRes = nn.resData[resIndexShift + i]; 
		
				double delta = lastRes - results[i - nn.lastLayerFirstIndex];
				nn.deltasData[resIndexShift + i] = delta;
			}
			break;
	}

	// Start with last weight, to go backwards during calculations, using next layer neurons.
	// In case of perceptron (no hidden layers) pointer below will be incorrect, but it will not be used anyway, because function will end by for condition.
	double *nextLayerLastWeight = nn.weights + nn.inputLayerNeuronsCount * nn.neuronsPerHiddenLayer + (nn.hiddenLayersCount - 1) * nn.neuronsPerHiddenLayer * nn.neuronsPerHiddenLayer + nn.neuronsPerHiddenLayer * nn.outputLayerNeuronsCount - 1;
	double *nextLayerLastWeightForCurrentNeuron = nextLayerLastWeight;
	for(; i >= nn.inputLayerNeuronsCount; i--) {
		struct Neuron *n = &(nn.net[i]);

		int indexInLayer = n->index;

		// Look up and sum next layer neuron deltas, multiplied by weights.
		int nextLayerFirstIndex = nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * n->layer;

		int nextLayerNeuronsCount;
		if(n->layer == nn.hiddenLayersCount) nextLayerNeuronsCount = nn.outputLayerNeuronsCount;
		else nextLayerNeuronsCount = nn.neuronsPerHiddenLayer;

		double *weight = nextLayerLastWeightForCurrentNeuron;
		double dErrorBydSigma = 0;
		for(int k = nextLayerFirstIndex + nextLayerNeuronsCount - 1; k >= nextLayerFirstIndex; k--) {
			//struct Neuron nextLayerNeuron = nn.net[k];
			dErrorBydSigma += nn.deltasData[resIndexShift + k] * (*weight); //nextLayerNeuron.weights[indexInLayer];
			weight -= nn.neuronsPerHiddenLayer;
		}

		if(indexInLayer == 0) {
			nextLayerLastWeight -= nextLayerNeuronsCount * nn.neuronsPerHiddenLayer;
			nextLayerLastWeightForCurrentNeuron = nextLayerLastWeight;
		}
		else nextLayerLastWeightForCurrentNeuron -= 1;

		double lastRes = nn.resData[resIndexShift + i]; 
		double lastActivationDerivative = derivativeOfLastActivation(lastRes, n->aft);
		double delta = lastActivationDerivative * dErrorBydSigma;
		nn.deltasData[resIndexShift + i] = delta;
	}
}

// Whoever read this - sorry for quite messy code below - original readable version was sacrificed for performance. I was surprised by how much improvements can be achieved even just by moving variables initializations layers above, if possible.
void updateWeights(struct NeuralNetwork nn, int trainBlockSize) {
	/*if(trainBlockSize < 1) {
		fprintf(stderr, "\ntrain size < 1\n");
	}*/
	double trainCoeff = nn.baseTrainCoeff * nn.trainCoeffCurrentDecreaser;
	double trainBlockCoeff = 1.0 / trainBlockSize;
	double invertedSamplesNumber = 1.0;
	if(nn.trainSamplesTotalAmount > 0) invertedSamplesNumber = 1.0 / nn.trainSamplesTotalAmount;
	double l2RegularizationReducedBySamplesNumber = nn.l2RegularizationParameter * invertedSamplesNumber;

	int previousLayerFirstIndex = 0;
	int previousLayerNeuronsCount = nn.inputLayerNeuronsCount;
	int currentLayerNeuronsCount = nn.hiddenLayersCount > 0 ? nn.neuronsPerHiddenLayer : nn.outputLayerNeuronsCount;
	for(int layer = 1; layer <= nn.hiddenLayersCount + 1; layer++) {
		bool firstHidden = layer == 1;
		for(int index = 0; index < currentLayerNeuronsCount; index++) {
			int neuronTotalIndex = nn.inputLayerNeuronsCount + (layer - 1) * nn.neuronsPerHiddenLayer + index;
			int weightsFirstIndex = index * previousLayerNeuronsCount + !firstHidden * (nn.neuronsPerHiddenLayer * nn.inputLayerNeuronsCount + (layer - 2) * nn.neuronsPerHiddenLayer * nn.neuronsPerHiddenLayer);

			double *weight = nn.weights + weightsFirstIndex;
			double *velocities = nn.weightsVelocities + weightsFirstIndex;
			double *bias = nn.bias + neuronTotalIndex - nn.inputLayerNeuronsCount;
			double *biasVelocities = nn.biasVelocities + neuronTotalIndex - nn.inputLayerNeuronsCount;
			double sumOfDeltas = 0;
			double *resultsOfPreviousLayer = nn.resData + previousLayerFirstIndex;//It's content is sigma in formulas.

			for(int k = 0; k < previousLayerNeuronsCount; k++) {
				double sumOfWeightDeltas = 0;
				for(int block = 0; block < trainBlockSize; block++) {
					int trainBlockShift = nn.neuronsCount * block;
					sumOfWeightDeltas += (*(resultsOfPreviousLayer + trainBlockShift)) * nn.deltasData[trainBlockShift + neuronTotalIndex];// This multiplication is gradient of weight.
				}
				double velocity = -trainCoeff * (sumOfWeightDeltas * trainBlockCoeff + l2RegularizationReducedBySamplesNumber * weight[k]);
				if(nn.weightsMomentum > 0) {
					velocity += nn.weightsMomentum * velocities[k];
					velocities[k] = velocity;
				}
				weight[k] += velocity;
				resultsOfPreviousLayer++;
			}
			for(int block = 0; block < trainBlockSize; block++) {
				int trainBlockShift = nn.neuronsCount * block;
				sumOfDeltas += nn.deltasData[trainBlockShift + neuronTotalIndex];
			}
			double biasVelocity = -trainCoeff * sumOfDeltas * trainBlockCoeff;
			if(nn.weightsMomentum > 0) {
				biasVelocity += nn.weightsMomentum * (*biasVelocities);
				(*biasVelocities) = biasVelocity;
			}
			*bias += biasVelocity;
		}

		// Set info for next layer, therefore do it in the layer before last for neurons count;
		previousLayerNeuronsCount = currentLayerNeuronsCount;
		if(layer == nn.hiddenLayersCount) {
			currentLayerNeuronsCount = nn.outputLayerNeuronsCount;
			// Here must be setting of separate trainCoeff for output layer then implemented.
		}
		previousLayerFirstIndex = nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * (layer - 1);
	}
}

// Used in simple gradient descent below.
void train(struct NeuralNetwork nn, double *inputs, double *outputs, int maxCycles) {
	for(int i = 0; i < maxCycles; i++) {
		calculate(nn, inputs, 0);

		calculateDeltas(nn, outputs, 0);

		updateWeights(nn, 1);
	}
}

// Train by one sample at the time, cycling all of them.
void trainByGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, double costFunctionToStop, int maxCycles) {
	// Array of indexes to shuffle examples before each training cycle.
	int* indexes = malloc(examplesQuantity * sizeof(int));
	printf("\nindexes for shuffle created\n");
	for(int i = 0; i < examplesQuantity; i++) {
		indexes[i] = i;
	}
	printf("\nindexes set\n");
	double *input = malloc(inputSize * sizeof(double));
	double *output = malloc(outputSize * sizeof(double));

	for(int c = 0; c < maxCycles; c++) {
		printf("\ngroup train cycle %d", c);
		if(logsFile != NULL) fprintf(logsFile, "\ngroup train cycle %d", c);

		double cycleCost = 0;
		shuffle(indexes, examplesQuantity);
		for(int i = 0; i < examplesQuantity; i++) {
			//printf("\ngroup train cycle %d example %d", c, i);

			for(int k = 0; k < inputSize; k++) {
				input[k] = inputs[inputSize * indexes[i] + k];
			}
			for(int k = 0; k < outputSize; k++) {
				output[k] = outputs[outputSize * indexes[i] + k];
			}
/*
			for(int k = 0; k < 784; k++) {
				printf(" %03.0f", input[k]);
				if(k % 28 == 27) printf("\n");
			}
*/
			train(nn, input, output, 1);

			double cost = costFunction(nn, output, 1, i % 10000 == 0 ? logsFile : NULL);
			cycleCost += cost;
		}
		cycleCost /= examplesQuantity;
		printf("\ncycleCost %f\n", cycleCost);
		if(cycleCost < costFunctionToStop) break;
	}
	free(input);
	free(output);
		
	free(indexes);
}

// Train by all samples at once.
void trainByBatchGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, double costFunctionToStop, int maxCycles) {
	double *input = malloc(inputSize * sizeof(double));
	double *output = malloc(outputSize * sizeof(double));
	for(int c = 0; c < maxCycles; c++) {
		if(logsFile != NULL) fprintf(logsFile, "\ngroup together train cycle %d", c);
		for(int i = 0; i < examplesQuantity; i++) {
			for(int k = 0; k < inputSize; k++) {
				input[k] = inputs[inputSize * i + k];
			}
			for(int k = 0; k < outputSize; k++) {
				output[k] = outputs[outputSize * i + k];
			}

			calculate(nn, input, i);
			calculateDeltas(nn, output, i);
		}
		printNetworkInExistingFILE(nn, logsFile);

		double cost = costFunction(nn, outputs, examplesQuantity, logsFile);
		if(cost < costFunctionToStop) {
			free(input);
			free(output);
			return;
		}

		updateWeights(nn, examplesQuantity);
	}
	free(input);
	free(output);
}

// Train by small batch of samples at once, reshuffling after all batches was processed in current cycle.
void trainByMiniBatchStochasticGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, int maxCycles, int batchSize, double (*netCorrectness)()) {
	// Array of indexes to shuffle examples before each training cycle.
	int* indexes = malloc(examplesQuantity * sizeof(int));
	for(int i = 0; i < examplesQuantity; i++) {
		indexes[i] = i;
	}

	int internalCycles = examplesQuantity / batchSize;
	int lastBatchSize = examplesQuantity % batchSize;
	int totalInternalCycles = internalCycles + (lastBatchSize > 0 ? 1 : 0);

	double *input = malloc(inputSize * sizeof(double));
	double *output = malloc(outputSize * sizeof(double));
	double *batchOutputs = malloc(outputSize * batchSize * sizeof(double));

	int lastImprovementCycle = 0;
	double lastImprovementCorrectness = 0;

	for(int c = 0; c < maxCycles; c++) {
		if(logsFile != NULL) fprintf(logsFile, "\ngroup together train cycle %d", c);

		shuffle(indexes, examplesQuantity);

		double cycleCost = 0;

		for(int internalCycle = 0; internalCycle < totalInternalCycles; internalCycle++) {
			int samplesCount = batchSize;
			if(lastBatchSize > 0 && internalCycle == totalInternalCycles - 1) {
				samplesCount = lastBatchSize;
			}

			int startIndex = internalCycle * batchSize;
			for(int i = 0; i < samplesCount; i++) {
				for(int k = 0; k < inputSize; k++) {
					input[k] = inputs[inputSize * indexes[startIndex + i] + k];
				}
				for(int k = 0; k < outputSize; k++) {
					output[k] = outputs[outputSize * indexes[startIndex + i] + k];
					batchOutputs[i * outputSize + k] = output[k];
				}

				calculate(nn, input, i);
				calculateDeltas(nn, output, i);
			}

			updateWeights(nn, samplesCount);
			double batchCost = costFunction(nn, batchOutputs, samplesCount, logsFile);
			cycleCost += batchCost;
		}

		cycleCost /= totalInternalCycles;
		if(logsFile != NULL) fprintf(logsFile, "\ncycle %d cost function: %f\n", c, cycleCost);
		printf("\ncycle %d cost function: %f\n", c, cycleCost);

		if(netCorrectness != NULL) {
			double correctness = (*netCorrectness)();
			//printf("\ncorrectness by train data: %f, by test data: %f\n", correctness.trainRes, correctness.testRes);
			printf("\ncorrectness by test data: %f\n", correctness);
			if(nn.noImprovementsEpochsLimit > 0) {
				if(correctness > lastImprovementCorrectness) {
					lastImprovementCorrectness = correctness;
					lastImprovementCycle = c;
				} else if(c - lastImprovementCycle > nn.noImprovementsEpochsLimit) {
					if(nn.trainCoeffCurrentDecreaser > nn.trainCoeffDecreaserLimit) {
						nn.trainCoeffCurrentDecreaser *= 0.5;
						lastImprovementCycle = c;
						printf("\nDecrease learning rate.\n");
					} else {
						printf("\nExit because of no improvements for too long.\n");
						free(indexes);
						free(batchOutputs);
						free(input);
						free(output);
						return;
					}
				}
			}
		}
	}

	free(batchOutputs);
	free(input);
	free(output);
	free(indexes);
}

// For MNIST and other type-recognizing things, there one max value in output define result.
int testNetworkByEvalData(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity) {
	int inputSize = nn.inputLayerNeuronsCount;
	int outputSize = nn.outputLayerNeuronsCount;
	int correctCounter = 0;
	for(int i = 0; i < examplesQuantity; i++) {
		double *input = inputs + inputSize * i;
		calculate(nn, input, 0);

		double maxRes = -1;
		int maxResIndex = 0;

		double *output = outputs + outputSize * i;
		for(int k = 0; k < outputSize; k++) {
			double res = output[k];
			if(res > maxRes) {
				maxRes = res;
				maxResIndex = k;
			}
		}
		int correctResIndex = maxResIndex;

		maxRes = -1;
		maxResIndex = 0;
		for(int k = 0; k < outputSize; k++) {
			double res = nn.resData[nn.lastLayerFirstIndex + k];
			if(res > maxRes) {
				maxRes = res;
				maxResIndex = k;
			}
		}
		if(maxResIndex == correctResIndex) correctCounter++;
	}
	printf("\nEvaluation, correct data: %d/%d\n", correctCounter, examplesQuantity);

	return correctCounter;
}

