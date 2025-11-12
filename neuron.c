#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

FILE *logsFile;

void setupLogs() {
	logsFile = NULL;//fopen("logs.txt", "w");
	//logsFile = fopen("/dev/null", "w");
}

void closeLogs() {
	if(logsFile != NULL) fclose(logsFile);
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

enum ActivationFunctionType {
	sigmoid,
	ReLU
};

struct Neuron {
	int layer;
	int index;
	int weightsCount;
	double* weights;
	double bias;
	enum ActivationFunctionType aft;
	// for last results, that will be used afterwards in training
	int resCount;
	double *results;
	double *deltas;
};

double propagation(struct Neuron neuron, double* inputs, int inputsCount) {
	/*if(neuron.weightsCount != inputsCount) {
		fprintf(stderr, "Inputs and weights different quantities");
		exit(1);
	}*/
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
}

double activation(struct Neuron neuron, double* inputs, int inputsCount) {
	/*if(neuron.weightsCount != inputsCount) {
		fprintf(stderr, "Inputs and weights different quantities");
		exit(1);
	}*/

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

double derivativeOfLastActivation(struct Neuron neuron, int resIndex) {
	/*if(resIndex < 0) {
		fprintf(stderr, "\nresult index < 0\n");
		exit(1);
	}*/
	double lastRes = neuron.results[resIndex];
	switch(neuron.aft) {
		case sigmoid:
			return lastRes * (1 - lastRes);
		case ReLU:
			return lastRes <= 0 ? 0 : 1;
	}
}

//TODO Will change after other activation functions are introduced.
double derivativeOfActivation(struct Neuron neuron, double *inputs, int resIndex) {
	// for sigmoid it is simple, but for other function must change
	return derivativeOfLastActivation(neuron, resIndex);
}

struct NeuralNetwork {
	struct Neuron *net;
	int neuronsCount;
	int inputLayerNeuronsCount;
	int outputLayerNeuronsCount;
	int hiddenLayersCount;
	int neuronsPerHiddenLayer;
	int trainBlockSize;
	double *lastCosts;
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

	nn->net = calloc(neuronsCount, sizeof(struct Neuron));
	nn->lastCosts = calloc(trainBlockSize, sizeof(double));

	// input layer
	for(int i = 0; i < inputLayerNeuronsCount; i++) {
		nn->net[i].layer = 0;
		nn->net[i].index = i;
		nn->net[i].weightsCount = 1;
		nn->net[i].weights = malloc(sizeof(double));
		nn->net[i].weights[0] = 1;
		nn->net[i].bias = 0;
		nn->net[i].aft = aft;
		nn->net[i].resCount = trainBlockSize;
		nn->net[i].results = calloc(trainBlockSize, sizeof(double));
		nn->net[i].deltas = calloc(trainBlockSize, sizeof(double));
	}

	// output layer
	for(int i = 0; i < outputLayerNeuronsCount; i++) {
		int il = i + inputLayerNeuronsCount + hiddenLayersCount * neuronsPerHiddenLayer;
		nn->net[il].layer = 1 + hiddenLayersCount;
		nn->net[il].index = i;
		nn->net[il].weightsCount = neuronsPerHiddenLayer;
		nn->net[il].weights = malloc(neuronsPerHiddenLayer * sizeof(double));
		for(int k = 0; k < neuronsPerHiddenLayer; k++) {
			nn->net[il].weights[k] = randomf(-1, 1);
		}
		nn->net[il].bias = randomf(-2, 2);
		nn->net[il].aft = aft;
		nn->net[il].resCount = trainBlockSize;
		nn->net[il].results = calloc(trainBlockSize, sizeof(double));
		nn->net[il].deltas = calloc(trainBlockSize, sizeof(double));
	}

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
			nn->net[il].weights = malloc(previousLayerNeuronsCount * sizeof(double));
			for(int k = 0; k < previousLayerNeuronsCount; k++) {
				nn->net[il].weights[k] = randomf(-1, 1);
			}
			nn->net[il].bias = randomf(-2, 2);
			nn->net[il].aft = aft;
			nn->net[il].resCount = trainBlockSize;
			nn->net[il].results = calloc(trainBlockSize, sizeof(double));
			nn->net[il].deltas = calloc(trainBlockSize, sizeof(double));
		}
	}

	return nn;
}

void destroyNetwork(struct NeuralNetwork **nn) {
	for(int i = 0; i < (**nn).neuronsCount; i++) {
		free((**nn).net[i].weights);
		free((**nn).net[i].results);
		free((**nn).net[i].deltas);
	}
	free((**nn).net);
	free((**nn).lastCosts);
	free(*nn);
	*nn = NULL;
}

void calculate(struct NeuralNetwork nn, double *inputs, int resIndex) {
	if(resIndex < 0) {
		fprintf(stderr, "\nresult index < 0\n");
		exit(1);
	}
	for(int i = 0; i < nn.inputLayerNeuronsCount; i++) {
		nn.net[i].results[resIndex] = inputs[i];
	}

	int maxPrevLayerCount = nn.inputLayerNeuronsCount;
	if(maxPrevLayerCount < nn.neuronsPerHiddenLayer) maxPrevLayerCount = nn.neuronsPerHiddenLayer;
	double *prevLayerRes = malloc(sizeof(double) * maxPrevLayerCount);

	for(int i = nn.inputLayerNeuronsCount; i < nn.neuronsCount; i++) {
		struct Neuron n = nn.net[i];

		int previousLayerFirstIndex;
		if(n.layer == 1) {
			previousLayerFirstIndex = 0;
		} else {
			previousLayerFirstIndex = nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * (n.layer - 2);
		}

		for(int k = 0; k < n.weightsCount; k++) {
			prevLayerRes[k] = nn.net[previousLayerFirstIndex + k].results[resIndex];
		}

		double res = activation(n, prevLayerRes, n.weightsCount);

		nn.net[i].results[resIndex] = res;
	}
	free(prevLayerRes);
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
	for(int i = 0; i < nn.neuronsCount; i++) {
		struct Neuron n = nn.net[i];

		int layerWidth = n.layer == 0 ? 9 : 7 * n.weightsCount + 16 + widthReserve;
		for(int k = 0; k < n.index; k++) {
			printf("\n");
		}
		if(n.index == 0) {
			printf("\r");
		}

		//int xCoord = n.layer * (perLayer + betweenLayers);
		if(lastLayer < n.layer) {
			lastLayer = n.layer;
			lastEolXCoord += layerWidth;
		}
		int xCoord = lastEolXCoord - layerWidth; 

		// move cursor right
		printf("\x1b[%dC", xCoord);
		if(n.layer == 0) {
			printf("%.4f", n.results[0]);
		} else {
			for(int w = 0; w < n.weightsCount; w++) {
				printf("%.4f ", n.weights[w]);
			}
			printf("%.4f ", n.bias);
			//printf("%d %d %d", i, xCoord, n.weightsCount);
			printf("%.4f %.3f", n.results[0], n.deltas[0]);

		}
		if(n.index > 0) {
			// move cursor up specified number of lines
			printf("\x1b[%dA", n.index);
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
	int neuronsCount = nn.inputLayerNeuronsCount + nn.outputLayerNeuronsCount + nn.hiddenLayersCount * nn.neuronsPerHiddenLayer;
	int i = 0;
	while(level < maxLevel) {
		struct Neuron n = nn.net[i];

		char neuronInfo[30];
		if(n.layer == 0) {
			fprintf(logsFile, "\nneuron ");
			if(n.resCount > 0) {
				for(int r = 0; r < n.resCount; r++) {
					fprintf(logsFile, "res%d: %.4f ", r, n.results[r]);
				}
			}
		} else {
			fprintf(logsFile, "neuron layer %d index %d weights: ", n.layer, n.index);
			for(int w = 0; w < n.weightsCount; w++) {
				fprintf(logsFile, "%.4f ", n.weights[w]);
			}
			fprintf(logsFile, "bias: %.4f ", n.bias);
			//printf("%d %d %d", i, xCoord, n.weightsCount);
			if(n.resCount > 0) {
				for(int r = 0; r < n.resCount; r++) {
					fprintf(logsFile, "res%d: %.4f delta%d: %.3f ", r, n.results[r], r, n.deltas[r]);
				}
			}
		}

		fprintf(logsFile, "     ");

		if(i < nn.inputLayerNeuronsCount) {
			if(level < nn.neuronsPerHiddenLayer) {
				i += nn.inputLayerNeuronsCount;
			} else if(level < nn.outputLayerNeuronsCount) {
				i += nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * nn.hiddenLayersCount;
			} else if(i == nn.inputLayerNeuronsCount - 1) break;
			else i = neuronsCount;// make sure that level increase in only one place
		}
		else if(i < nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * nn.hiddenLayersCount) {
			// if after this go beyound last layer, then it means it is shorter than hidden ones
			i += nn.neuronsPerHiddenLayer;
		}
		else if(level < nn.inputLayerNeuronsCount || level < nn.neuronsPerHiddenLayer) i += nn.outputLayerNeuronsCount;// next level afterwards
		else i = neuronsCount;// make sure that level increase in only one place

		if(i >= neuronsCount) {
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
double costFunction(struct NeuralNetwork nn, double *desiredOutputs, int trainBlockSize, FILE *logsOutput) {
	if(trainBlockSize < 1) {
		fprintf(stderr, "\ntrainSize < 1\n");
		exit(1);
	}
	// calculate cost function as 1/2 * sum(errorPerOutputNeuron^2)
	double cost = 0;
	int lastLayerFirstIndex = nn.inputLayerNeuronsCount + nn.hiddenLayersCount * nn.neuronsPerHiddenLayer;

	for(int resIndex = 0; resIndex < trainBlockSize; resIndex++) {
		double sampleCost = 0;
		for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
			sampleCost += pow(nn.net[lastLayerFirstIndex + i].results[resIndex] - desiredOutputs[nn.outputLayerNeuronsCount * resIndex + i], 2);
		}
		sampleCost *= 0.5;
		nn.lastCosts[resIndex] = sampleCost;
		cost += sampleCost;
	}
	cost /= trainBlockSize;

	if(logsOutput != NULL) {
		/*fprintf(logsOutput, "\ninputs:");
		for(int block = 0; block < trainBlockSize; block++) {
			fprintf(logsOutput, "\n    inputs%d:", block);
			for(int i = 0; i < nn.inputLayerNeuronsCount; i++) {
				fprintf(logsOutput, " %f", nn.net[i].results[block]);
			}
		}*/
	
		fprintf(logsOutput, "\nresults:");
		for(int block = 0; block < trainBlockSize; block++) {
			fprintf(logsOutput, "\n    results%d:", block);
			for(int i = 0; i < nn.outputLayerNeuronsCount; i++) {
				fprintf(logsOutput, " %f", nn.net[lastLayerFirstIndex + i].results[block]);
			}
		}
	
		fprintf(logsOutput, "\nwe need:");//text like this to make desired results right under real ones
		for(int block = 0; block < trainBlockSize; block++) {
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
	if(resIndex < 0) {
		fprintf(stderr, "\nresIndex < 0\n");
		exit(1);
	}
	// neurons
	int lastLayerFirstIndex = nn.inputLayerNeuronsCount + nn.hiddenLayersCount * nn.neuronsPerHiddenLayer;
	for(int i = lastLayerFirstIndex + nn.outputLayerNeuronsCount - 1; i >= nn.inputLayerNeuronsCount; i--) {
		struct Neuron *n = &(nn.net[i]);
		double lastActivationDerivative = derivativeOfLastActivation(*n, resIndex);

		double dErrorBydSigma = 0;
		if(n->layer == nn.hiddenLayersCount + 1) {// last layer
			dErrorBydSigma = n->results[resIndex] - results[i - lastLayerFirstIndex];
		} else {
			int indexInLayer;
			if(i < nn.inputLayerNeuronsCount) indexInLayer = i;
			else indexInLayer = (i - nn.inputLayerNeuronsCount) % nn.neuronsPerHiddenLayer;

			// look and sum next layer neuron deltas, multiplied by weights
			int nextLayerFirstIndex = nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * n->layer;

			int nextLayerNeuronsCount;
			if(n->layer == nn.hiddenLayersCount) nextLayerNeuronsCount = nn.outputLayerNeuronsCount;
			else nextLayerNeuronsCount = nn.neuronsPerHiddenLayer;

			for(int k = nextLayerFirstIndex; k < nextLayerFirstIndex + nextLayerNeuronsCount; k++) {
				struct Neuron nextLayerNeuron = nn.net[k];
				dErrorBydSigma += nextLayerNeuron.deltas[resIndex] * nextLayerNeuron.weights[indexInLayer];
			}
		}
		double delta = lastActivationDerivative * dErrorBydSigma;
		n->deltas[resIndex] = delta;
	}
}

void updateWeights(struct NeuralNetwork nn, int trainBlockSize) {
	if(trainBlockSize < 1) {
		fprintf(stderr, "\ntrain size < 1\n");
	}
	double trainKoeff = nn.net[0].aft == sigmoid ? 2.8 : 0.09;

	for(int i = nn.inputLayerNeuronsCount; i < nn.inputLayerNeuronsCount + nn.outputLayerNeuronsCount + nn.hiddenLayersCount * nn.neuronsPerHiddenLayer; i++) {
		struct Neuron *n = &(nn.net[i]);
		for(int k = 0; k < n->weightsCount; k++) {
			int previousLayerFirstIndex;
			if(n->layer == 1) previousLayerFirstIndex = 0;
			else previousLayerFirstIndex = nn.inputLayerNeuronsCount + nn.neuronsPerHiddenLayer * (n->layer - 2);
			
			double gradientOfWeight = 0;
			for(int block = 0; block < trainBlockSize; block++) {
				double sigma = nn.net[previousLayerFirstIndex + k].results[block];
				gradientOfWeight += sigma * n->deltas[block];
			}
			gradientOfWeight /= trainBlockSize;
			
			n->weights[k] -= trainKoeff * gradientOfWeight;
		}
		// for bias formula remain the same, except derivative of propagation function by bias is 1, since its constant
		double sumOfDeltas = 0;
		for(int block = 0; block < trainBlockSize; block++) {
			sumOfDeltas += n->deltas[block];
		}
		n->bias -= trainKoeff * sumOfDeltas / trainBlockSize;
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
			double res = nn.net[lastLayerFirstIndex + k].results[0];
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

	//trainByGradientDescent(*nn, inputs, outputs, mnistTrainImages.count, 784, 10, costFuncToStop, maxTrainCycles);
	trainByMiniBatchStochasticGradientDescent(*nn, inputs, outputs, mnistTrainImages.count, 784, 10, costFuncToStop, maxTrainCycles, trainBlockSize, testNetworkByMNISTDataForFunctionParam);

	//testNetworkByMNISTData(*nn, inputsTest, outputsTest, mnistTestImages.count);

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
