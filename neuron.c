#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <semaphore.h>

FILE *logsFile = NULL;

void setupLogs() {
	//logsFile = fopen("logs.txt", "w");
	//logsFile = fopen("/dev/null", "w");
}

void closeLogs() {
	if(logsFile != NULL) fclose(logsFile);
}

int sign(int x) {
	return (x > 0) - (x < 0);
}

double randomf(double from, double to) {
	double interval = to - from;
	assert(interval > 0);
	int rand = random();
	double fraction = (double)rand / RAND_MAX;
	return from + fraction * interval;
}

void shuffle(int* array, int length) {
	if(length < 1) return;
	for(int i = 0; i < length; i++) {
		int index = rand() % length;
		int element = array[i];
		array[i] = array[index];
		array[index] = element;
	}
}

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

enum ActivationFunctionType {
	sigmoid,
	ReLU
};

struct Neuron {
	int layer;
	int index;
	int weightsCount;
	enum ActivationFunctionType aft;
};
/*
double propagation(struct Neuron neuron, double* inputs, int inputsCount) {
//	if(neuron.weightsCount != inputsCount) {
//		fprintf(stderr, "Inputs and weights different quantities");
//		exit(1);
//	}
	double res = 0;
	int i = 0;

	// Surprisingly good optimization - MNIST 10 cycles train and check time drop from 1:45 to 1:15 from this alone. Increasing hardcode more dont provide benefits.
	for(; i < inputsCount - 5; i += 5) {
		res += neuron.weights[i] * inputs[i] +
			neuron.weights[i+1] * inputs[i+1] +
			neuron.weights[i+2] * inputs[i+2] +
			neuron.weights[i+3] * inputs[i+3] +
			neuron.weights[i+4] * inputs[i+4];
	}
	for(; i < inputsCount; i++) {
		res += neuron.weights[i] * inputs[i];
	}

	res += neuron.bias;
	return res;
}*/
/*
double activation(struct Neuron neuron, double* inputs, int inputsCount) {
//	if(neuron.weightsCount != inputsCount) {
//		fprintf(stderr, "Inputs and weights different quantities");
//		exit(1);
//	}

	double arg = propagation(neuron, inputs, inputsCount);
	double res;
	switch(neuron.aft) {
		case sigmoid:
			res = 1 / (1 + pow(M_E, -arg));
			break;
		case ReLU:
			res = arg < 0 ? 0 : arg;
			break;
	}

	return res;
}
*/
double activation(double propagation, enum ActivationFunctionType aft) {
	switch(aft) {
		case sigmoid:
			return 1 / (1 + exp(-propagation));
		case ReLU:
			return propagation < 0 ? 0 : propagation;
	}
}

double derivativeOfLastActivation(double lastRes, enum ActivationFunctionType aft) {
	switch(aft) {
		case sigmoid:
			return lastRes * (1 - lastRes);
		case ReLU:
			return lastRes <= 0 ? 0 : 1;
	}
}

//TODO Will change after other activation functions are introduced.
/*double derivativeOfActivation(struct Neuron neuron, double *inputs, int resIndex) {
	// for sigmoid it is simple, but for other function must change
	return derivativeOfLastActivation(neuron, resIndex);
}*/

struct NeuralNetwork {
	struct Neuron *net;
	int neuronsCount;
	int inputLayerNeuronsCount;
	int outputLayerNeuronsCount;
	int hiddenLayersCount;
	int neuronsPerHiddenLayer;
	int trainBlockSize;
	int lastLayerFirstIndex;
	double *lastCosts;
	// Results go like this: allResults_block1, allResults_block2 ... allResults_last
	double *resData;
	double *deltasData;
	double *weights;
	double *bias;
};

struct NeuralNetwork *createNetwork(int inputLayerNeuronsCount, int outputLayerNeuronsCount, int hiddenLayersCount, int neuronsPerHiddenLayer, int trainBlockSize, enum ActivationFunctionType aft) {
	int neuronsCount = inputLayerNeuronsCount + outputLayerNeuronsCount + hiddenLayersCount * neuronsPerHiddenLayer;
	struct NeuralNetwork *nn = malloc(sizeof(struct NeuralNetwork));

	nn->neuronsCount = neuronsCount;
	nn->inputLayerNeuronsCount = inputLayerNeuronsCount;
	nn->outputLayerNeuronsCount = outputLayerNeuronsCount;
	nn->hiddenLayersCount = hiddenLayersCount;
	nn->neuronsPerHiddenLayer = neuronsPerHiddenLayer;
	nn->trainBlockSize = trainBlockSize;
	nn->lastLayerFirstIndex = neuronsCount - outputLayerNeuronsCount;

	nn->net = calloc(neuronsCount, sizeof(struct Neuron));
	nn->lastCosts = calloc(trainBlockSize, sizeof(double));
	nn->resData = calloc(trainBlockSize * neuronsCount, sizeof(double));
	nn->deltasData = calloc(trainBlockSize * neuronsCount, sizeof(double));
	// First hidden layer has number of weights equal to input layer neurons number multiplied to number of neurons in layer itself, the same logic applied to other hidden layers and output layer.
	// WARNING it work only if hidden layers exist - percepton will fail.
	nn->weights = malloc((inputLayerNeuronsCount * neuronsPerHiddenLayer + (hiddenLayersCount - 1) * neuronsPerHiddenLayer * neuronsPerHiddenLayer + neuronsPerHiddenLayer * outputLayerNeuronsCount) * sizeof(double));
	nn->bias = malloc((neuronsCount - inputLayerNeuronsCount) * sizeof(double));

	// input layer
	for(int i = 0; i < inputLayerNeuronsCount; i++) {
		nn->net[i].layer = 0;
		nn->net[i].index = i;
		nn->net[i].weightsCount = 1;
		nn->net[i].aft = aft;
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
			for(int k = 0; k < previousLayerNeuronsCount; k++) {
				double w = randomf(-1, 1);
				nn->weights[weightIndex] = w;
				weightIndex++;
			}
			double bias = randomf(-2, 2);
			nn->bias[il - inputLayerNeuronsCount] = bias;
			nn->net[il].aft = aft;
		}
	}

	// output layer
	for(int i = 0; i < outputLayerNeuronsCount; i++) {
		int il = i + nn->lastLayerFirstIndex;
		nn->net[il].layer = 1 + hiddenLayersCount;
		nn->net[il].index = i;
		nn->net[il].weightsCount = neuronsPerHiddenLayer;
		for(int k = 0; k < neuronsPerHiddenLayer; k++) {
			double w = randomf(-1, 1);
			nn->weights[weightIndex] = w;
			weightIndex++;
		}
		double bias = randomf(-2, 2);
		nn->bias[il - inputLayerNeuronsCount] = bias;
		nn->net[il].aft = aft;
	}

	return nn;
}

void destroyNetwork(struct NeuralNetwork **nn) {
	free((**nn).net);
	free((**nn).resData);
	free((**nn).deltasData);
	free((**nn).weights);
	free((**nn).bias);
	free((**nn).lastCosts);
	free(*nn);
	*nn = NULL;
}

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

void* processActivation(void *pad) {
	struct ProcessActivationData *data = (struct ProcessActivationData *)pad;
	double *weights = data->weights;
	double *bias = data->bias;
	double *resData = data->resData;
	for(int i = 0; i < data->neuronsCount; i++) {
		double propagation = dotProduct(weights, data->prevLayerResults, data->weightsNumber);
		propagation += *bias;
		weights += data->weightsNumber;
		bias++;
		*resData = activation(propagation, data->aft);
		resData++;
	}
}

void *processActivationQueue1(void *args) {
	int prevType;
	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &prevType);

	while(1) {
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
void calculate(struct NeuralNetwork nn, double *inputs, int resIndex) {
	/*if(resIndex < 0) {
		fprintf(stderr, "\nresult index < 0\n");
		exit(1);
	}*/
	int resIndexShift = nn.neuronsCount * resIndex;
	for(int i = 0; i < nn.inputLayerNeuronsCount; i++) {
		nn.resData[resIndexShift + i] = inputs[i];
	}
/*
	// Process first hidden layer separately for performance.
	int i = nn.inputLayerNeuronsCount;
	for(; i < nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer; i++) {
		struct Neuron n = nn.net[i];
		double *prevLayerRes = nn.resData + resIndexShift;
		double res = activation(n, prevLayerRes, n.weightsCount);

		nn.resData[resIndexShift + i] = res;
	}
	// From second hidden layer till the end.
	for(; i < nn.neuronsCount; i++) {
		struct Neuron n = nn.net[i];

		int previousLayerFirstIndex = nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * (n.layer - 2);
		double *prevLayerRes = nn.resData + resIndexShift + previousLayerFirstIndex;
		double res = activation(n, prevLayerRes, n.weightsCount);

		nn.resData[resIndexShift + i] = res;
	}
*/
	// First hidden layer.
	double *weights = nn.weights;
	double *bias = nn.bias;
	double *prevLayerRes = nn.resData + resIndexShift;
	double *resData = prevLayerRes + nn.inputLayerNeuronsCount;
/*
	// Threading first crude attempt. Slower than one thread. Clearly contain some error, that causes calculation mistakes in rare cases, but didn't found for now.
	int threadNumber = 2;
	pthread_t threads[2];
	int neuronsPerThread = nn.neuronsPerHiddenLayer / threadNumber;
	int lastThreadNeurons = nn.neuronsPerHiddenLayer % threadNumber;
	// To not go beyound threadNumber, add rest of neurons to the last operation.
	if(lastThreadNeurons > 0) lastThreadNeurons += neuronsPerThread;
	for(int threadId = 0; threadId < threadNumber; threadId++) {
		pthread_t thread;
		struct ProcessActivationData data;
		data.weightsNumber = nn.inputLayerNeuronsCount;
		data.aft = nn.net[nn.inputLayerNeuronsCount].aft;
		data.prevLayerResults = prevLayerRes;
		data.weights = weights + threadId * neuronsPerThread * nn.inputLayerNeuronsCount;
		data.bias = bias + threadId * neuronsPerThread;
		data.resData = resData + threadId * neuronsPerThread;
		if(threadId == threadNumber - 1 && lastThreadNeurons > 0) {
			data.neuronsCount = lastThreadNeurons;
		} else {
			data.neuronsCount = neuronsPerThread;
		}
		pthread_create(&threads[threadId], NULL, processActivation, &data);
	}
	for(int threadId = 0; threadId < threadNumber; threadId ++) {
		pthread_join(threads[threadId], NULL);
	}

	weights += nn.inputLayerNeuronsCount * nn.neuronsPerHiddenLayer;
	bias += nn.neuronsPerHiddenLayer;
	resData += nn.neuronsPerHiddenLayer;
*/
/*
	// Threading second version - work significantly faster, than one thread, didn't notice rare errors, like in previous version.
	struct ProcessActivationData data1, data2, data3;
	data1.weightsNumber = nn.inputLayerNeuronsCount;
	data1.aft = nn.net[nn.inputLayerNeuronsCount].aft;
	data1.prevLayerResults = prevLayerRes;
	data1.weights = weights;
	data1.bias = bias;
	data1.resData = resData;
	data1.neuronsCount = nn.neuronsPerHiddenLayer / 4;

	data2.weightsNumber = nn.inputLayerNeuronsCount;
	data2.aft = nn.net[nn.inputLayerNeuronsCount].aft;
	data2.prevLayerResults = prevLayerRes;
	data2.weights = weights + data1.neuronsCount * nn.inputLayerNeuronsCount;
	data2.bias = bias + data1.neuronsCount;
	data2.resData = resData + data1.neuronsCount;
	data2.neuronsCount = data1.neuronsCount;

	data3.weightsNumber = nn.inputLayerNeuronsCount;
	data3.aft = nn.net[nn.inputLayerNeuronsCount].aft;
	data3.prevLayerResults = prevLayerRes;
	data3.weights = weights + 2 * data1.neuronsCount * nn.inputLayerNeuronsCount;
	data3.bias = bias + 2 * data1.neuronsCount;
	data3.resData = resData + 2 * data1.neuronsCount;
	data3.neuronsCount = data1.neuronsCount;

	int previousNeurons = 3 * data1.neuronsCount;
	weights += nn.inputLayerNeuronsCount * previousNeurons;
	bias += previousNeurons;
	resData += previousNeurons;

	pad1 = &data1;
	pad2 = &data2;
	pad3 = &data3;

	for(int i = previousNeurons; i < nn.neuronsPerHiddenLayer; i++) {
		double propagation = dotProduct(weights, prevLayerRes, nn.inputLayerNeuronsCount);
		propagation += *bias;
		weights += nn.inputLayerNeuronsCount;
		bias++;
		*resData = activation(propagation, nn.net[nn.inputLayerNeuronsCount + i].aft);
		resData++;
	}

	sem_wait(&sem1);
	sem_wait(&sem2);
	sem_wait(&sem3);

	// Other hidden layers
	prevLayerRes += nn.inputLayerNeuronsCount;
	for(int layer = 1; layer < nn.hiddenLayersCount; layer++) {
		data1.weightsNumber = nn.neuronsPerHiddenLayer;
		data1.aft = nn.net[nn.inputLayerNeuronsCount].aft;
		data1.prevLayerResults = prevLayerRes;
		data1.weights = weights;
		data1.bias = bias;
		data1.resData = resData;
		data1.neuronsCount = nn.neuronsPerHiddenLayer / 4;
	
		data2.weightsNumber = nn.neuronsPerHiddenLayer;
		data2.aft = nn.net[nn.inputLayerNeuronsCount].aft;
		data2.prevLayerResults = prevLayerRes;
		data2.weights = weights + data1.neuronsCount * nn.neuronsPerHiddenLayer;
		data2.bias = bias + data1.neuronsCount;
		data2.resData = resData + data1.neuronsCount;
		data2.neuronsCount = data1.neuronsCount;
	
		data3.weightsNumber = nn.neuronsPerHiddenLayer;
		data3.aft = nn.net[nn.inputLayerNeuronsCount].aft;
		data3.prevLayerResults = prevLayerRes;
		data3.weights = weights + 2 * data1.neuronsCount * nn.neuronsPerHiddenLayer;
		data3.bias = bias + 2 * data1.neuronsCount;
		data3.resData = resData + 2 * data1.neuronsCount;
		data3.neuronsCount = data1.neuronsCount;
	
		previousNeurons = 3 * data1.neuronsCount;
		weights += nn.neuronsPerHiddenLayer * previousNeurons;
		bias += previousNeurons;
		resData += previousNeurons;
	
		pad1 = &data1;
		pad2 = &data2;
		pad3 = &data3;

		for(int i = previousNeurons; i < nn.neuronsPerHiddenLayer; i++) {
			double propagation = dotProduct(weights, prevLayerRes, nn.neuronsPerHiddenLayer);
			propagation += *bias;
			weights += nn.neuronsPerHiddenLayer;
			bias++;
			*resData = activation(propagation, nn.net[nn.inputLayerNeuronsCount + i].aft);
			resData++;
		}
		sem_wait(&sem1);
		sem_wait(&sem2);
		sem_wait(&sem3);

		prevLayerRes += nn.neuronsPerHiddenLayer;
	}

	// Output layer.
	data1.weightsNumber = nn.neuronsPerHiddenLayer;
	data1.aft = nn.net[nn.neuronsCount - 1].aft;
	data1.prevLayerResults = prevLayerRes;
	data1.weights = weights;
	data1.bias = bias;
	data1.resData = resData;
	data1.neuronsCount = nn.outputLayerNeuronsCount / 4;

	data2.weightsNumber = nn.neuronsPerHiddenLayer;
	data2.aft = nn.net[nn.neuronsCount - 1].aft;
	data2.prevLayerResults = prevLayerRes;
	data2.weights = weights + data1.neuronsCount * nn.neuronsPerHiddenLayer;
	data2.bias = bias + data1.neuronsCount;
	data2.resData = resData + data1.neuronsCount;
	data2.neuronsCount = data1.neuronsCount;

	data3.weightsNumber = nn.neuronsPerHiddenLayer;
	data3.aft = nn.net[nn.neuronsCount - 1].aft;
	data3.prevLayerResults = prevLayerRes;
	data3.weights = weights + 2 * data1.neuronsCount * nn.neuronsPerHiddenLayer;
	data3.bias = bias + 2 * data1.neuronsCount;
	data3.resData = resData + 2 * data1.neuronsCount;
	data3.neuronsCount = data1.neuronsCount;

	previousNeurons = 3 * data1.neuronsCount;
	weights += nn.neuronsPerHiddenLayer * previousNeurons;
	bias += previousNeurons;
	resData += previousNeurons;

	pad1 = &data1;
	pad2 = &data2;
	pad3 = &data3;

	for(int i = previousNeurons; i < nn.outputLayerNeuronsCount; i++) {
		double propagation = dotProduct(weights, prevLayerRes, nn.neuronsPerHiddenLayer);
		propagation += *bias;
		weights += nn.neuronsPerHiddenLayer;
		bias++;
		*resData = activation(propagation, nn.net[nn.neuronsCount - 1].aft);
		resData++;
	}

	sem_wait(&sem1);
	sem_wait(&sem2);
	sem_wait(&sem3);
*/

	// More readable threading version.
	struct ProcessActivationData data1, data2, data3;
	for(int layer = 0; layer <= nn.hiddenLayersCount; layer++) {
		int weightsNumber = layer == 0 ? nn.inputLayerNeuronsCount : nn.neuronsPerHiddenLayer;
		enum ActivationFunctionType aft = layer == nn.hiddenLayersCount ? nn.net[nn.neuronsCount - 1].aft : nn.net[nn.inputLayerNeuronsCount].aft;
		int neuronsCount = layer == nn.hiddenLayersCount ? nn.outputLayerNeuronsCount : nn.neuronsPerHiddenLayer;

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
	
		int previousNeurons = 3 * data1.neuronsCount;
		weights += previousNeurons * weightsNumber;
		bias += previousNeurons;
		resData += previousNeurons;
	
		pad1 = &data1;
		pad2 = &data2;
		pad3 = &data3;

		for(int i = previousNeurons; i < neuronsCount; i++) {
			double propagation = dotProduct(weights, prevLayerRes, weightsNumber);
			propagation += *bias;
			weights += weightsNumber;
			bias++;
			*resData = activation(propagation, aft);
			resData++;
		}
		sem_wait(&sem1);
		sem_wait(&sem2);
		sem_wait(&sem3);

		prevLayerRes += weightsNumber;
	}

/*
	// One thread.
	for(int i = 0; i < nn.neuronsPerHiddenLayer; i++) {
		double propagation = dotProduct(weights, prevLayerRes, nn.inputLayerNeuronsCount);
		propagation += *bias;
		weights += nn.inputLayerNeuronsCount;
		bias++;
		*resData = activation(propagation, nn.net[nn.inputLayerNeuronsCount + i].aft);
		resData++;
	}

	// Other hidden layers
	prevLayerRes += nn.inputLayerNeuronsCount;
	for(int layer = 1; layer < nn.hiddenLayersCount; layer ++) {
		for(int i = 0; i < nn.neuronsPerHiddenLayer; i++) {
			double propagation = dotProduct(weights, prevLayerRes, nn.neuronsPerHiddenLayer);
			propagation += *bias;
			weights += nn.neuronsPerHiddenLayer;
			bias++;
			*resData = activation(propagation, nn.net[nn.inputLayerNeuronsCount + i].aft);
			resData++;
		}
		prevLayerRes += nn.neuronsPerHiddenLayer;
	}

	// Output layer.
	for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
		double propagation = dotProduct(weights, prevLayerRes, nn.neuronsPerHiddenLayer);
		propagation += *bias;
		weights += nn.neuronsPerHiddenLayer;
		bias++;
		*resData = activation(propagation, nn.net[nn.neuronsCount - 1].aft);
		resData++;
	}
*/
}

// for small networks only and one result
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

void printNetworkInFile(struct NeuralNetwork nn) {
	int maxLevel = nn.inputLayerNeuronsCount;
	if(nn.outputLayerNeuronsCount > maxLevel) maxLevel = nn.outputLayerNeuronsCount;
	if(nn.neuronsPerHiddenLayer > maxLevel) maxLevel = nn.neuronsPerHiddenLayer;

	int level = 0;
	int i = 0;
	while(level < maxLevel) {
		struct Neuron n = nn.net[i];
		char neuronInfo[30];
		if(n.layer == 0) {
			fprintf(logsFile, "\nneuron ");
			for(int r = 0; r < nn.trainBlockSize; r++) {
				fprintf(logsFile, "res%d: %.4f ", r, nn.resData[nn.neuronsCount * r + i]);
			}
		} else {
			fprintf(logsFile, "neuron layer %d index %d weights: ", n.layer, n.index);
			int firstWeightIndex;
			if(n.layer == 1) {
				firstWeightIndex = nn.inputLayerNeuronsCount * n.index;
			} else {
				firstWeightIndex = nn.neuronsPerHiddenLayer * nn.inputLayerNeuronsCount + (n.layer - 2) * nn.neuronsPerHiddenLayer * nn.neuronsPerHiddenLayer + n.index * nn.neuronsPerHiddenLayer;
			}
			for(int w = 0; w < n.weightsCount; w++) {
				fprintf(logsFile, "%.4f ", nn.weights[firstWeightIndex + w]);
			}
			fprintf(logsFile, "bias: %.4f ", nn.bias[i - nn.inputLayerNeuronsCount]);
			//printf("%d %d %d", i, xCoord, n.weightsCount);
			for(int r = 0; r < nn.trainBlockSize; r++) {
				fprintf(logsFile, "res%d: %.4f delta%d: %.3f ", r, nn.resData[nn.neuronsCount * r + i], r, nn.deltasData[nn.neuronsCount * r + i]);
			}
		}

		fprintf(logsFile, "     ");

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
			fprintf(logsFile, "\n");
			// find first layer with enough neurons
			if(level < nn.inputLayerNeuronsCount) i = level;
			else if(level < nn.neuronsPerHiddenLayer) i = level + nn.inputLayerNeuronsCount;
			else i = nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * nn.hiddenLayersCount + level;
		}
	}
}

// cost function
double costFunction(struct NeuralNetwork nn, double *desiredOutputs, int samplesCount, FILE *logsOutput) {
	/*if(samplesCount < 1) {
		fprintf(stderr, "\ntrainSize < 1\n");
		exit(1);
	}*/
	// calculate cost function as 1/2 * sum(errorPerOutputNeuron^2)
	double cost = 0;

	for(int resIndex = 0; resIndex < samplesCount; resIndex++) {
		double sampleCost = 0;
		for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
			sampleCost += pow(nn.resData[nn.neuronsCount * resIndex + nn.lastLayerFirstIndex + i] - desiredOutputs[nn.outputLayerNeuronsCount * resIndex + i], 2);
		}
		sampleCost *= 0.5;
		nn.lastCosts[resIndex] = sampleCost;
		cost += sampleCost;
	}
	cost /= samplesCount;

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

// add bias in formula after main mechanics work
void calculateDeltas(struct NeuralNetwork nn, double *results, int resIndex) {
	/*if(resIndex < 0) {
		fprintf(stderr, "\nresIndex < 0\n");
		exit(1);
	}*/
	// neurons
	int i = nn.neuronsCount - 1;
	int resIndexShift = nn.neuronsCount * resIndex;

	// Last layer process separately for a bit of performance.
	for(; i >= nn.lastLayerFirstIndex; i--) {
		struct Neuron *n = &(nn.net[i]);
		double lastRes = nn.resData[resIndexShift + i]; 

		double dErrorBydSigma = lastRes - results[i - nn.lastLayerFirstIndex];
		double lastActivationDerivative = derivativeOfLastActivation(lastRes, n->aft);
		double delta = lastActivationDerivative * dErrorBydSigma;
		nn.deltasData[resIndexShift + i] = delta;
	}

	// Start with last weight, to go backwards during calculations, using next layer neurons.
	double *nextLayerLastWeight = nn.weights + nn.inputLayerNeuronsCount * nn.neuronsPerHiddenLayer + (nn.hiddenLayersCount - 1) * nn.neuronsPerHiddenLayer * nn.neuronsPerHiddenLayer + nn.neuronsPerHiddenLayer * nn.outputLayerNeuronsCount - 1;
	double *nextLayerLastWeightForCurrentNeuron = nextLayerLastWeight;
	for(; i >= nn.inputLayerNeuronsCount; i--) {
		struct Neuron *n = &(nn.net[i]);

		int indexInLayer = n->index;

		// look and sum next layer neuron deltas, multiplied by weights
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

void updateWeights(struct NeuralNetwork nn, int trainBlockSize) {
	/*if(trainBlockSize < 1) {
		fprintf(stderr, "\ntrain size < 1\n");
	}*/
	double trainKoeff;
	switch(nn.net[0].aft) {
		case sigmoid:
			trainKoeff = 2.8;
			break;
		case ReLU:
			trainKoeff = 0.09;
			break;
	}
	double trainBlockCoeff = 1.0 / trainBlockSize;

	int previousLayerFirstIndex = 0;
	int previousLayerNeuronsCount = nn.inputLayerNeuronsCount;
	int currentLayerNeuronsCount = nn.neuronsPerHiddenLayer;
	for(int layer = 1; layer <= nn.hiddenLayersCount + 1; layer++) {
		bool firstHidden = layer == 1;
		for(int index = 0; index < currentLayerNeuronsCount; index++) {
			int neuronTotalIndex = nn.inputLayerNeuronsCount + (layer - 1) * nn.neuronsPerHiddenLayer + index;
			double *weight = nn.weights + index * previousLayerNeuronsCount + !firstHidden * (nn.neuronsPerHiddenLayer * nn.inputLayerNeuronsCount + (layer - 2) * nn.neuronsPerHiddenLayer * nn.neuronsPerHiddenLayer);
			double *bias = nn.bias + neuronTotalIndex - nn.inputLayerNeuronsCount;
			double sumOfDeltas = 0;
			for(int block = 0; block < trainBlockSize; block++) {
				int trainBlockShift = nn.neuronsCount * block;
				double *resultsOfPreviousLayer = nn.resData + trainBlockShift + previousLayerFirstIndex;//It's content is sigma in formulas.
				double deltaReducedByTrainBlocks = nn.deltasData[trainBlockShift + neuronTotalIndex] * trainBlockCoeff;
				for(int k = 0; k < previousLayerNeuronsCount; k++) {
					weight[k] -= trainKoeff * (*resultsOfPreviousLayer) * deltaReducedByTrainBlocks;// Multiplication of last two is gradient of weight, divided by number of training blocks.
					resultsOfPreviousLayer++;
				}
				sumOfDeltas += nn.deltasData[trainBlockShift + neuronTotalIndex];
			}
			*bias -= trainKoeff * sumOfDeltas * trainBlockCoeff;
		}
		previousLayerNeuronsCount = currentLayerNeuronsCount;
		if(layer == nn.hiddenLayersCount) currentLayerNeuronsCount = nn.outputLayerNeuronsCount;
		previousLayerFirstIndex = nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * (layer - 1);
	}
}

void train(struct NeuralNetwork nn, double *inputs, double *outputs, int maxCycles) {
	for(int i = 0; i < maxCycles; i++) {
		calculate(nn, inputs, 0);

		//printNetworkInFile(nn);
		calculateDeltas(nn, outputs, 0);

		updateWeights(nn, 1);
	}
}

// train by one sample at the time, cycling all of them
void trainByGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, double costFunctionToStop, int maxCycles) {
	// array of indexes to shuffle examples before each training cycle
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
		fprintf(logsFile, "\ngroup train cycle %d", c);

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
		//printNetworkInFile(nn);
		cycleCost /= examplesQuantity;
		printf("\ncycleCost %f\n", cycleCost);
		if(cycleCost < costFunctionToStop) break;
	}
	free(input);
	free(output);
		
	free(indexes);
}

// train by all samples at once 
void trainByBatchGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, double costFunctionToStop, int maxCycles) {
	double *input = malloc(inputSize * sizeof(double));
	double *output = malloc(outputSize * sizeof(double));
	for(int c = 0; c < maxCycles; c++) {
		fprintf(logsFile, "\ngroup together train cycle %d", c);
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
		printNetworkInFile(nn);

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

// train by small batch of samples at once, reshuffling after all batches was processed in current cycle
void trainByMiniBatchStochasticGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, double costFunctionToStop, int maxCycles, int batchSize, double (*mnistCorrectness)()) {
	// array of indexes to shuffle examples before each training cycle
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

	for(int c = 0; c < maxCycles; c++) {
		//fprintf(logsFile, "\ngroup together train cycle %d", c);

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
		//printNetworkInFile(nn);

		cycleCost /= totalInternalCycles;
		//fprintf(logsFile, "\ncycle %d cost function: %f\n", c, cycleCost);
		printf("\ncycle %d cost function: %f\n", c, cycleCost);

		if(mnistCorrectness != NULL) {
			double correctness = (*mnistCorrectness)();
			printf("\ncorrectness %f\n", correctness);
		}

		if(cycleCost < costFunctionToStop) {
			free(indexes);
			free(batchOutputs);
			free(input);
			free(output);
			return;
		}
	}

	free(batchOutputs);
	free(input);
	free(output);
	free(indexes);
}

void testXOR() {
	int maxTrainCycles = 3000;
	double costFuncToStop = 0.015;

	int trainBlockSize = 4;
	struct NeuralNetwork *nn = createNetwork(2, 1, 1, 2, trainBlockSize, sigmoid);
	printf("\nnetwork created\n");

	double inputs[] = {1, 1};
	double outputs[] = { 0 };

	double inputs1[] = {1, 0};
	double outputs1[] = { 1 };

	double inputs2[] = {0, 1};
	double outputs2[] = { 1 };

	double inputs3[] = {0, 0};
	double outputs3[] = { 0 };

	double groupInputs[] = {
				1, 1,
       				1, 0,
				0, 1,
				0, 0
	};
	double groupOutputs[] = {
				0,
				1,
				1,
				0
	};
	trainByGradientDescent(*nn, groupInputs, groupOutputs, 4, 2, 1, costFuncToStop, maxTrainCycles);
	//trainByBatchGradientDescent(*nn, groupInputs, groupOutputs, 4, 2, 1, costFuncToStop, maxTrainCycles);
	//trainByMiniBatchStochasticGradientDescent(*nn, groupInputs, groupOutputs, 4, 2, 1, costFuncToStop, maxTrainCycles, 2, NULL);

	printf("\ntest xor:\n");

	calculate(*nn, inputs, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs, 1, stdout);

	calculate(*nn, inputs1, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs1, 1, stdout);

	calculate(*nn, inputs2, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs2, 1, stdout);

	calculate(*nn, inputs3, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs3, 1, stdout);
	printNetworkInFile(*nn);

	destroyNetwork(&nn);
}

void testOR() {
	int maxTrainCycles = 1000;
	double costFuncToStop = 0.015;

	int trainBlockSize = 4;
	struct NeuralNetwork *nn = createNetwork(2, 1, 1, 2, trainBlockSize, sigmoid);
	printf("\nnetwork created\n");

	double inputs[] = {1, 1};
	double outputs[] = { 1 };

	double inputs1[] = {1, 0};
	double outputs1[] = { 1 };

	double inputs2[] = {0, 1};
	double outputs2[] = { 1 };

	double inputs3[] = {0, 0};
	double outputs3[] = { 0 };

	double groupInputs[] = {
				1, 1,
       				1, 0,
				0, 1,
				0, 0
	};
	double groupOutputs[] = {
				1,
				1,
				1,
				0
	};
	//trainByGradientDescent(*nn, groupInputs, groupOutputs, 4, 2, 1, costFuncToStop, maxTrainCycles);
	trainByBatchGradientDescent(*nn, groupInputs, groupOutputs, 4, 2, 1, costFuncToStop, maxTrainCycles);
	//trainByMiniBatchStochasticGradientDescent(*nn, groupInputs, groupOutputs, 4, 2, 1, costFuncToStop, maxTrainCycles, 2, NULL);

	printf("\ntest or:\n");

	calculate(*nn, inputs, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs, 1, stdout);

	calculate(*nn, inputs1, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs1, 1, stdout);

	calculate(*nn, inputs2, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs2, 1, stdout);

	calculate(*nn, inputs3, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs3, 1, stdout);

	destroyNetwork(&nn);
}

void testAND() {
	int maxTrainCycles = 1000;
	double costFuncToStop = 0.015;

	int trainBlockSize = 4;
	struct NeuralNetwork *nn = createNetwork(2, 1, 1, 2, trainBlockSize, sigmoid);
	printf("\nnetwork created\n");

	double inputs[] = {1, 1};
	double outputs[] = { 1 };

	double inputs1[] = {1, 0};
	double outputs1[] = { 0 };

	double inputs2[] = {0, 1};
	double outputs2[] = { 0 };

	double inputs3[] = {0, 0};
	double outputs3[] = { 0 };

	double groupInputs[] = {
				1, 1,
       				1, 0,
				0, 1,
				0, 0
	};
	double groupOutputs[] = {
				1,
				0,
				0,
				0
	};
	//trainByGradientDescent(*nn, groupInputs, groupOutputs, 4, 2, 1, costFuncToStop, maxTrainCycles);
	//trainByBatchGradientDescent(*nn, groupInputs, groupOutputs, 4, 2, 1, costFuncToStop, maxTrainCycles);
	trainByMiniBatchStochasticGradientDescent(*nn, groupInputs, groupOutputs, 4, 2, 1, costFuncToStop, maxTrainCycles, 2, NULL);

	printf("\ntest and:\n");

	calculate(*nn, inputs, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs, 1, stdout);

	calculate(*nn, inputs1, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs1, 1, stdout);

	calculate(*nn, inputs2, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs2, 1, stdout);

	calculate(*nn, inputs3, 0);
	printNetwork(*nn);
	costFunction(*nn, outputs3, 1, stdout);

	destroyNetwork(&nn);
}

struct MNIST_Data {
	int dimensionsAmount;
	int *dimensions;
	int count;
	unsigned char *data;
};

void freeMNIST(struct MNIST_Data mnist) {
	free(mnist.dimensions);
	free(mnist.data);
}

int testNetworkByMNISTData(struct NeuralNetwork nn, double *inputs, unsigned char *outputs, int examplesQuantity) {
	int mnistSize = 784;
	int digits = 10;
	double *input = malloc(mnistSize * sizeof(double));
	int correctCounter = 0;
	for(int i = 0; i < examplesQuantity; i++) {
		for(int k = 0; k < mnistSize; k++) {
			input[k] = inputs[mnistSize * i + k];
		}

		calculate(nn, input, 0);

		int lastLayerFirstIndex = nn.inputLayerNeuronsCount + nn.hiddenLayersCount * nn.neuronsPerHiddenLayer;
		double maxRes = -1;
		int maxResIndex = 0;
		int correctResIndex = outputs[i];

		for(int k = 0; k < digits; k++) {
			double res = nn.resData[lastLayerFirstIndex + k];
			if(res > maxRes) {
				maxRes = res;
				maxResIndex = k;
			}
		}
		if(maxResIndex == correctResIndex) correctCounter++;
	}
	printf("\nmnist correct data: %d/%d\n", correctCounter, examplesQuantity);
	free(input);

	return correctCounter;
}

struct NeuralNetwork *globalValNetworkForMNISTSpecialTest;
double *globalValMNISTInputs;
unsigned char *globalValMNISTOutputs;
int globalValMNISTExamplesQuantity;

double testNetworkByMNISTDataForFunctionParam() {
	int correctAmount = testNetworkByMNISTData(*globalValNetworkForMNISTSpecialTest, globalValMNISTInputs, globalValMNISTOutputs, globalValMNISTExamplesQuantity);
	double res = ((double)correctAmount) / globalValMNISTExamplesQuantity;
	return res;
}

struct MNIST_Data readMNIST(char *fileName) {
	FILE *file = fopen(fileName, "rb");
	unsigned char mainInfoBuffer[4];
	int *dimensions = NULL;
	int samplesCount;
	unsigned char *data;
	fread(mainInfoBuffer, sizeof(mainInfoBuffer), 1, file);

	struct MNIST_Data mnist;

	int dimensionsAmount = mainInfoBuffer[3];
	if(dimensionsAmount > 0) {
		int dimensionsBufferSize = dimensionsAmount * sizeof(uint32_t);
		int sampleSize = 1;
		uint32_t *dimensionsBuffer = malloc(dimensionsBufferSize);
		uint32_t *allDimensions = malloc(dimensionsBufferSize);

		fread(dimensionsBuffer, dimensionsBufferSize, 1, file);
		for(int i = 0; i < dimensionsAmount; i++) {
			// Process big/little endians.
			allDimensions[i] = __builtin_bswap32(dimensionsBuffer[i]);
		}
		samplesCount = allDimensions[0];
		if(dimensionsAmount > 1) {
			dimensions = malloc((dimensionsAmount - 1) * sizeof(int));
			for(int i = 1; i < dimensionsAmount; i++) {
				dimensions[i - 1] = allDimensions[i];
				sampleSize *= dimensions[i - 1];
			}
		} else {
			dimensions = malloc(sizeof(int));
			dimensions[0] = 1;
		}
		free(allDimensions);
		free(dimensionsBuffer);

		int dataSize = sampleSize * samplesCount * sizeof(char);
		unsigned char *buf = malloc(dataSize);
		fread(buf, dataSize, 1, file);

		mnist.dimensionsAmount = dimensionsAmount < 2 ? 1 : dimensionsAmount - 1;
		mnist.dimensions = dimensions;
		mnist.count = samplesCount;
		mnist.data = buf;
/*
		for(int sample = samplesCount - 10; sample < samplesCount; sample++) {
			printf("\nsample %d\n", sample);
			for(int i = 0; i < 784; i++) {
				printf(" %03d", buf[sample * 784 + i]);
				if(i % 28 == 27) printf("\n");
			}
		}
*/
	} else {
		fclose(file);
		fprintf(stderr, "wrong mnist file format");
		exit(1);
	}
	fclose(file);

	return mnist;
}

void testMNIST() {
	// train data
	struct MNIST_Data mnistTrainImages = readMNIST("train-images.idx3-ubyte");
	struct MNIST_Data mnistTrainLabels = readMNIST("train-labels.idx1-ubyte");
/*
	for(int sample = 0; sample < 10; sample++) {
		printf("\nsample %d contain %d\n", sample, mnistTrainLabels.data[sample]);
		for(int i = 0; i < 784; i++) {
			printf(" %03d", mnistTrainImages.data[sample * 784 + i]);
			if(i % 28 == 27) printf("\n");
		}
	}
*/

	double *inputs = malloc(mnistTrainImages.count * 784 * sizeof(double));
	double *outputs = calloc(mnistTrainLabels.count * 10, sizeof(double));
	for(int i = 0; i < mnistTrainLabels.count; i++) {
		outputs[i * 10 + mnistTrainLabels.data[i]] = 1;
		for(int k = 0; k < 784; k++) {
			inputs[i * 784 + k] = ((double)mnistTrainImages.data[i * 784 + k]) / 255.0;
		}
	}
	
	printf("\ntrain inputs set\n");

	// test data
	struct MNIST_Data mnistTestImages = readMNIST("t10k-images.idx3-ubyte");
	struct MNIST_Data mnistTestLabels = readMNIST("t10k-labels.idx1-ubyte");
/*
	for(int sample = 0; sample < 10; sample++) {
		printf("\nsample %d contain %d\n", sample, mnistTestLabels.data[sample]);
		for(int i = 0; i < 784; i++) {
			printf(" %03d", mnistTestImages.data[sample * 784 + i]);
			if(i % 28 == 27) printf("\n");
		}
	}
*/
	double *inputsTest = malloc(mnistTestImages.count * 784 * sizeof(double));
	for(int i = 0; i < mnistTestLabels.count; i++) {
		for(int k = 0; k < 784; k++) {
			inputsTest[i * 784 + k] = ((double)mnistTestImages.data[i * 784 + k]) / 255.0;
		}
	}

	printf("\ntest inputs set\n");

	int trainBlockSize = 10;
	struct NeuralNetwork *nn = createNetwork(784, 10, 1, 30, trainBlockSize, sigmoid);
	printf("\nnetwork created\n");

	double costFuncToStop = 0.02;
	int maxTrainCycles = 10;

	globalValNetworkForMNISTSpecialTest = nn;
	globalValMNISTInputs = inputsTest;
	globalValMNISTOutputs = mnistTestLabels.data;
	globalValMNISTExamplesQuantity = mnistTestImages.count;


	sem_init(&sem1, 0, 0);
	sem_init(&sem2, 0, 0);
	sem_init(&sem3, 0, 0);
	pthread_create(&thread1, NULL, processActivationQueue1, NULL);
	pthread_create(&thread2, NULL, processActivationQueue2, NULL);
	pthread_create(&thread3, NULL, processActivationQueue3, NULL);

	//trainByGradientDescent(*nn, inputs, outputs, mnistTrainImages.count, 784, 10, costFuncToStop, maxTrainCycles);
	trainByMiniBatchStochasticGradientDescent(*nn, inputs, outputs, mnistTrainImages.count, 784, 10, costFuncToStop, maxTrainCycles, trainBlockSize, testNetworkByMNISTDataForFunctionParam);

	//testNetworkByMNISTData(*nn, inputsTest, outputsTest, mnistTestImages.count);

	pthread_cancel(thread1);
	pthread_cancel(thread2);
	pthread_cancel(thread3);
	sem_destroy(&sem1);
	sem_destroy(&sem2);
	sem_destroy(&sem3);

	free(inputs);
	free(outputs);
	free(inputsTest);
	destroyNetwork(&nn);
	freeMNIST(mnistTrainImages);
	freeMNIST(mnistTrainLabels);
	freeMNIST(mnistTestImages);
	freeMNIST(mnistTestLabels);
}

int main() {
	srandom(time(NULL));

	setupLogs();

	//testXOR();
	//testOR();
	//testAND();

	testMNIST();

	closeLogs();

	return 0;
}
