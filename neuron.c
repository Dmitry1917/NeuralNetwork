#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

FILE *logsFile = NULL;
bool useThreading = false;

void setupLogs() {
	//logsFile = fopen("logs.txt", "w");
	//logsFile = fopen("/dev/null", "w");
	//logsFile = stdout;
}

void closeLogs() {
	if(logsFile != NULL) fclose(logsFile);
}

int sign(int x) {
	return (x > 0) - (x < 0);
}

double randomUniform(double from, double to) {
	double interval = to - from;
	assert(interval > 0);
	int rand = random();
	double fraction = (double)rand / RAND_MAX;
	return from + fraction * interval;
}

double randomUniformForBoxMullerMethod() {// Return value in (0;1] semiopen interval.
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

// TODO Make better ways for combinations of cost functions and activation ones, to prevent accidently set incompatibles.
enum ActivationFunctionType {
	sigmoid,
	ReLU,
	softmax
};

enum CostFunctionType {
	square,
	crossEntropy,// Only use with sigmoid output layer.
	logLikehood// Only use with softmax output layer.
};

double activation(double propagation, enum ActivationFunctionType aft) {
	switch(aft) {
		case sigmoid:
			return 1 / (1 + exp(-propagation));
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
		case ReLU:
			return lastRes <= 0 ? 0 : 1;
		case softmax:
			printf("\nNot supposed use softmax here.\n");
			exit(1);

	}
}

//TODO Will change after other activation functions are introduced.
/*double derivativeOfActivation(struct Neuron neuron, double *inputs, int resIndex) {
	// for sigmoid it is simple, but for other function must change
	return derivativeOfLastActivation(neuron, resIndex);
}*/

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
	int lastLayerFirstIndex;
	double *lastCosts;
	// Results go like this: allResults_block1, allResults_block2 ... allResults_last
	double *resData;
	double *deltasData;
	double *weights;
	double *bias;
};

/* All info based on MNIST tests. trainCoeff for square cft and all sigmoids is 2.8. If use crossEntropy 0.5. softmax and cft logLikehood 0.05. ReLU for hidden layers demands decrease it to 0.1 and below if its higher.*/
struct NeuralNetwork *createNetwork(int inputLayerNeuronsCount, int outputLayerNeuronsCount, int hiddenLayersCount, int neuronsPerHiddenLayer, int trainBlockSize, enum ActivationFunctionType aftHidden, enum ActivationFunctionType aftOutput, enum CostFunctionType cft, double trainCoeff) {
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
	nn->baseTrainCoeff = trainCoeff;
	nn->l2RegularizationParameter = 0;
	nn->trainSamplesTotalAmount = 0;

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
			double deviation = 1.0 / sqrt(previousLayerNeuronsCount);
			for(int k = 0; k < previousLayerNeuronsCount; k++) {
				double w = randomGauss(0, deviation);
				//double w = randomUniform(-1, 1);
				nn->weights[weightIndex] = w;
				weightIndex++;
			}
			double bias = randomGauss(0, 1);
			//double bias = randomUniform(0, 1);
			nn->bias[il - inputLayerNeuronsCount] = bias;
			nn->net[il].aft = aftHidden;
		}
	}

	// output layer
	for(int i = 0; i < outputLayerNeuronsCount; i++) {
		int il = i + nn->lastLayerFirstIndex;
		nn->net[il].layer = 1 + hiddenLayersCount;
		nn->net[il].index = i;
		nn->net[il].weightsCount = neuronsPerHiddenLayer;
		double deviation = 1.0 / sqrt(neuronsPerHiddenLayer);
		for(int k = 0; k < neuronsPerHiddenLayer; k++) {
			double w = randomGauss(0, deviation);
			//double w = randomUniform(-1, 1);
			nn->weights[weightIndex] = w;
			weightIndex++;
		}
		double bias = randomGauss(0, 1);
		//double bias = randomUniform(0, 1);
		nn->bias[il - inputLayerNeuronsCount] = bias;
		nn->net[il].aft = aftOutput;
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

#define nanPreventionLimit 0.001
// cost function
double costFunction(struct NeuralNetwork nn, double *desiredOutputs, int samplesCount, FILE *logsOutput) {
	/*if(samplesCount < 1) {
		fprintf(stderr, "\ntrainSize < 1\n");
		exit(1);
	}*/
	// calculate cost function as 1/2 * sum(errorPerOutputNeuron^2)
	double cost = 0;

	switch(nn.cft) {
		case square:
			for(int resIndex = 0; resIndex < samplesCount; resIndex++) {
				double sampleCost = 0;
				for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
					sampleCost += pow(nn.resData[nn.neuronsCount * resIndex + nn.lastLayerFirstIndex + i] - desiredOutputs[nn.outputLayerNeuronsCount * resIndex + i], 2);
				}
				sampleCost *= 0.5;
				nn.lastCosts[resIndex] = sampleCost;
				cost += sampleCost;
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
				nn.lastCosts[resIndex] = sampleCost;
				cost += sampleCost;
			}
			break;
		case logLikehood:
			// This can be used only with activation functions, that lead to final results resemble probability distribution and desired results being 0 and 1 only.
			for(int resIndex = 0; resIndex < samplesCount; resIndex++) {
				double sampleCost = 0;
				for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
					double desired = desiredOutputs[nn.outputLayerNeuronsCount * resIndex + i];
					// Prevent floating point comparison issues again.
					if(desired > 0.999) {
						double real = nn.resData[nn.neuronsCount * resIndex + nn.lastLayerFirstIndex + i];
						sampleCost = -log(real);
						break;
					}
					if(i == nn.outputLayerNeuronsCount - 1) {
						printf("\nNot supposed to come here - desired results must have 1 if correct.\n");
						exit(1);
					}
				}
				nn.lastCosts[resIndex] = sampleCost;
				cost += sampleCost;
			}
			break;
	}
	if(nn.l2RegularizationParameter > 0) {
		int weightsCount = nn.inputLayerNeuronsCount * nn.neuronsPerHiddenLayer + (nn.hiddenLayersCount - 1) * nn.neuronsPerHiddenLayer * nn.neuronsPerHiddenLayer + nn.neuronsPerHiddenLayer * nn.outputLayerNeuronsCount;
		double weightsSquaresSum = 0;
		double *weights = nn.weights;
		for(int i = 0; i < weightsCount; i++) {
			weightsSquaresSum += *weights * *weights;
			weights++;
		}
		cost += nn.l2RegularizationParameter * weightsSquaresSum * 0.5;
	}
	// In all cases, there only part of samples is used, it will be later summed and divided by number of groups of samples, and that will made it like dividing all costs on all samples, like intended. To be completely fair - if total samples amount is not divisible on mini batch size, then there will be some error, but in all real cases (many samples, limited mini batch) it will be minor and inconsequential, because cost is used only for some control, not in calculations themselfs.
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
	double trainBlockCoeff = 1.0 / trainBlockSize;
	double invertedSamplesNumber = 1.0;
	if(nn.trainSamplesTotalAmount > 0) invertedSamplesNumber = 1.0 / nn.trainSamplesTotalAmount;

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
					weight[k] -= nn.baseTrainCoeff * (nn.l2RegularizationParameter * weight[k] * invertedSamplesNumber + (*resultsOfPreviousLayer) * deltaReducedByTrainBlocks);// Multiplication of last two is gradient of weight, divided by number of training blocks.
					resultsOfPreviousLayer++;
				}
				sumOfDeltas += nn.deltasData[trainBlockShift + neuronTotalIndex];
			}
			*bias -= nn.baseTrainCoeff * sumOfDeltas * trainBlockCoeff;
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

struct MNISTCheckResults {
	double trainRes;
	double testRes;
};

// train by small batch of samples at once, reshuffling after all batches was processed in current cycle
void trainByMiniBatchStochasticGradientDescent(struct NeuralNetwork nn, double *inputs, double *outputs, int examplesQuantity, int inputSize, int outputSize, double costFunctionToStop, int maxCycles, int batchSize, struct MNISTCheckResults (*mnistCorrectness)()) {
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
			struct MNISTCheckResults correctness = (*mnistCorrectness)();
			printf("\ncorrectness by train data: %f, by test data: %f\n", correctness.trainRes, correctness.testRes);
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
	struct NeuralNetwork *nn = createNetwork(2, 1, 1, 2, trainBlockSize, sigmoid, sigmoid, square, 2.5);
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
	struct NeuralNetwork *nn = createNetwork(2, 1, 1, 2, trainBlockSize, sigmoid, sigmoid, square, 2.5);
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
	trainByBatchGradientDescent(*nn, groupInputs, groupOutputs, 4, 2, 1, costFuncToStop, maxTrainCycles);

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
	struct NeuralNetwork *nn = createNetwork(2, 1, 1, 2, trainBlockSize, sigmoid, sigmoid, square, 2.5);
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

int testNetworkByMNISTData(struct NeuralNetwork nn, double *inputs, unsigned char *outputs, int examplesQuantity, bool isTrain) {
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
	printf("\nmnist correct data: %d/%d %b\n", correctCounter, examplesQuantity, isTrain);
	free(input);

	return correctCounter;
}

struct NeuralNetwork *globalValNetworkForMNISTSpecialTest;

double *globalValMNISTTrainInputs;
unsigned char *globalValMNISTTrainOutputs;
int globalValMNISTTrainExamplesQuantity;

double *globalValMNISTTestInputs;
unsigned char *globalValMNISTTestOutputs;
int globalValMNISTTestExamplesQuantity;

struct MNISTCheckResults testNetworkByMNISTDataForFunctionParam() {
	int correctAmountTrainData = testNetworkByMNISTData(*globalValNetworkForMNISTSpecialTest, globalValMNISTTrainInputs, globalValMNISTTrainOutputs, globalValMNISTTrainExamplesQuantity, true);
	int correctAmountTestData = testNetworkByMNISTData(*globalValNetworkForMNISTSpecialTest, globalValMNISTTestInputs, globalValMNISTTestOutputs, globalValMNISTTestExamplesQuantity, false);
	struct MNISTCheckResults res;
	res.trainRes = ((double)correctAmountTrainData) / globalValMNISTTrainExamplesQuantity;
	res.testRes = ((double)correctAmountTestData) / globalValMNISTTestExamplesQuantity;
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
	struct NeuralNetwork *nn = createNetwork(784, 10, 1, 30, trainBlockSize, sigmoid, sigmoid, crossEntropy, 0.5);
	printf("\nnetwork created\n");

	double costFuncToStop = 0.02;
	int maxTrainCycles = 10;

	globalValNetworkForMNISTSpecialTest = nn;
	globalValMNISTTestInputs = inputsTest;
	globalValMNISTTestOutputs = mnistTestLabels.data;
	globalValMNISTTestExamplesQuantity = mnistTestImages.count;

	globalValMNISTTrainInputs = inputs;
	globalValMNISTTrainOutputs = mnistTrainLabels.data;
	globalValMNISTTrainExamplesQuantity = mnistTrainImages.count;

	//nn->l2RegularizationParameter = 5.0;
	//nn->trainSamplesTotalAmount = mnistTrainImages.count;
	trainByMiniBatchStochasticGradientDescent(*nn, inputs, outputs, mnistTrainImages.count, 784, 10, costFuncToStop, maxTrainCycles, trainBlockSize, testNetworkByMNISTDataForFunctionParam);

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

	startThreading();

	//testXOR();
	//testOR();
	//testAND();

	testMNIST();

	stopThreading();

	closeLogs();

	// Wait for all subthreads to finish, to prevent valgrind from false leaks, but turns out it still gives some false positives on linux.
	//pthread_exit(NULL);
	return 0;
}
