#include "nn_NeuralNetwork.h"
#include "MNIST.h"
#include <stdlib.h>

void testXOR() {
	int maxTrainCycles = 3000;
	double costFuncToStop = 0.015;

	int trainBlockSize = 1;
	struct NeuralNetwork *nn = createNetwork(2, 1, 1, 2, trainBlockSize, sigmoid, sigmoid, square, 2.5, 0, 0, 0, NULL, NULL);
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
	printNetworkInFile(*nn, "xorNet.txt", false);
/*
	saveNetwork(*nn, "net.txt");
	struct NeuralNetwork *loadedNet = loadNetwork("net.txt", trainBlockSize, 2.5, 0, 0, 0);
	printf("\nLoaded network\n");
	calculate(*loadedNet, inputs, 0);
	printNetwork(*loadedNet);
	costFunction(*loadedNet, outputs, 1, stdout);
	destroyNetwork(&loadedNet);
*/
	destroyNetwork(&nn);
}

void testOR() {
	int maxTrainCycles = 1000;
	double costFuncToStop = 0.015;

	int trainBlockSize = 4;
	struct NeuralNetwork *nn = createNetwork(2, 1, 1, 2, trainBlockSize, sigmoid, sigmoid, square, 2.5, 0, 0, 0, NULL, NULL);
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

	int trainBlockSize = 2;
	struct NeuralNetwork *nn = createNetwork(2, 1, 1, 2, trainBlockSize, sigmoid, sigmoid, square, 2.5, 0, 0, 0, NULL, NULL);
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
	trainByMiniBatchStochasticGradientDescent(*nn, groupInputs, groupOutputs, 4, 2, 1, maxTrainCycles, trainBlockSize, NULL);

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

// Global values for network test during mini batch SGD.
struct NeuralNetwork *globalValNetworkForEvaluationTest;
double *globalValNetworkTestInputs;
double *globalValNetworkTestOutputs;
int globalValNetworkTestExamplesQuantity;

double evalNetworkByTestDataForFunctionParam() {
	int correctAmountTestData = testNetworkByEvalData(*globalValNetworkForEvaluationTest, globalValNetworkTestInputs, globalValNetworkTestOutputs, globalValNetworkTestExamplesQuantity);
	double res = ((double)correctAmountTestData) / globalValNetworkTestExamplesQuantity;
	return res;
}

void testAutoencoderMNIST() {
	// train data
	struct MNIST_Data mnistTrainImages = readMNIST("train-images.idx3-ubyte");
	struct MNIST_Data mnistTrainLabels = readMNIST("train-labels.idx1-ubyte");

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

	double *inputsTest = malloc(mnistTestImages.count * 784 * sizeof(double));
	double *outputsTest = calloc(mnistTestLabels.count * 10, sizeof(double));
	for(int i = 0; i < mnistTestLabels.count; i++) {
		outputsTest[i * 10 + mnistTestLabels.data[i]] = 1;
		for(int k = 0; k < 784; k++) {
			inputsTest[i * 784 + k] = ((double)mnistTestImages.data[i * 784 + k]) / 255.0;
		}
	}

	printf("\ntest inputs set\n");

	int trainBlockSize = 10;
	struct NeuralNetwork *nnAutoencoder = createNetwork(784, 784, 1, 100, trainBlockSize, ReLU, linear, square, 0.01, 0.0, 0, 0, NULL, NULL);
	printf("\nnetwork created\n");

	int maxTrainCycles = 10;

	// Check original chaos of encoder initialization.
	//calculate(*nnAutoencoder, inputs, 0);
	//costFunction(*nnAutoencoder, inputs, 1, stdout);

	//nnAutoencoder->l1RegularizationParameter = 0.0005;

	trainByMiniBatchStochasticGradientDescent(*nnAutoencoder, inputs, inputs, mnistTrainImages.count, 784, 784, maxTrainCycles, trainBlockSize, NULL);//evalNetworkByTestDataForFunctionParam);

	double *inputsPreprocessed = malloc(mnistTrainImages.count * 784 * sizeof(double));
	double *inputsTestPreprocessed = malloc(mnistTestImages.count * 784 * sizeof(double));

	int resultsFirstIndex = (*nnAutoencoder).lastLayerFirstIndex;
	for(int i = 0; i < mnistTrainLabels.count; i++) {
		calculate(*nnAutoencoder, inputs + i * 784, 0);
		for(int k = 0; k < 784; k++) {
			inputsPreprocessed[i * 784 + k] = (*nnAutoencoder).resData[resultsFirstIndex + k];
		}
	}
	for(int i = 0; i < mnistTestLabels.count; i++) {
		calculate(*nnAutoencoder, inputsTest + i * 784, 0);
		for(int k = 0; k < 784; k++) {
			inputsTestPreprocessed[i * 784 + k] = (*nnAutoencoder).resData[resultsFirstIndex + k];
		}
	}

	//printNetworkInFile(*nnAutoencoder, "netfile.txt", false);

	//calculate(*nnAutoencoder, inputs, 0);
	//costFunction(*nnAutoencoder, inputs, 1, stdout);

	printf("\nstart final network\n");
	maxTrainCycles = 10;

	struct NeuralNetwork *nn = createNetwork(784, 10, 1, 30, trainBlockSize, sigmoid, sigmoid, crossEntropy, 0.5, 0.0, 0, 0, NULL, NULL);
	globalValNetworkForEvaluationTest = nn;
	globalValNetworkTestInputs = inputsTestPreprocessed;
	globalValNetworkTestOutputs = outputsTest;
	globalValNetworkTestExamplesQuantity = mnistTestImages.count;

	nn->l2RegularizationParameter = 0.0005;
	nn->noImprovementsEpochsLimit = 5;
	nn->trainCoeffDecreaserLimit = 0.0625;
	nn->trainCoeffCurrentDecreaser = 1.0;
	trainByMiniBatchStochasticGradientDescent(*nn, inputsPreprocessed, outputs, mnistTrainImages.count, 784, 10, maxTrainCycles, trainBlockSize, evalNetworkByTestDataForFunctionParam);
/*
	saveNetwork(*nn, "net.txt");
	struct NeuralNetwork *loadedNet = loadNetwork("net.txt", trainBlockSize, 0.5, 0, 0, 0);
	printf("\nLoaded network\n");
	int correctAmountTestData = testNetworkByEvalData(*loadedNet, inputsTest, outputsTest, mnistTestImages.count);
	printf("\nCorrect test data: %d\n", correctAmountTestData);
	destroyNetwork(&loadedNet);
*/
	free(inputs);
	free(outputs);
	free(inputsTest);
	free(outputsTest);
	free(inputsPreprocessed);
	free(inputsTestPreprocessed);
	destroyNetwork(&nn);
	destroyNetwork(&nnAutoencoder);
	freeMNIST(mnistTrainImages);
	freeMNIST(mnistTrainLabels);
	freeMNIST(mnistTestImages);
	freeMNIST(mnistTestLabels);
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
	double *outputsTest = calloc(mnistTestLabels.count * 10, sizeof(double));
	for(int i = 0; i < mnistTestLabels.count; i++) {
		outputsTest[i * 10 + mnistTestLabels.data[i]] = 1;
		for(int k = 0; k < 784; k++) {
			inputsTest[i * 784 + k] = ((double)mnistTestImages.data[i * 784 + k]) / 255.0;
		}
	}

	printf("\ntest inputs set\n");

	int trainBlockSize = 10;
	struct NeuralNetwork *nn = createNetwork(784, 10, 1, 30, trainBlockSize, sigmoid, sigmoid, crossEntropy, 0.001, 0.9, 0.999, 0.00000001, NULL, NULL);
	printf("\nnetwork created\n");

	int maxTrainCycles = 30;

	globalValNetworkForEvaluationTest = nn;
	globalValNetworkTestInputs = inputsTest;
	globalValNetworkTestOutputs = outputsTest;
	globalValNetworkTestExamplesQuantity = mnistTestImages.count;

	//nn->l1RegularizationParameter = 0.0005;
	//nn->l2RegularizationParameter = 0.0005;
	//nn->decoupledWeightDecay = 0.0005;
	//nn->noImprovementsEpochsLimit = 5;
	//nn->trainCoeffDecreaserLimit = 0.0625;
	//nn->trainCoeffCurrentDecreaser = 1.0;
	nn->biasCorrectionInAdamOptimizer = true;
	trainByMiniBatchStochasticGradientDescent(*nn, inputs, outputs, mnistTrainImages.count, 784, 10, maxTrainCycles, trainBlockSize, evalNetworkByTestDataForFunctionParam);
/*
	printf("%.12f\n", nn->weightsGradientsMovingAverages[1000]);
	printf("%d\n", nn->iterationsOfWeightsUpdatesDone[0]);
	clearOptimizationsBuffers(*nn);
	printf("%.12f\n", nn->weightsGradientsMovingAverages[1000]);
	printf("%d\n", nn->iterationsOfWeightsUpdatesDone[0]);
*/
/*
	saveNetwork(*nn, "net.txt");
	struct NeuralNetwork *loadedNet = loadNetwork("net.txt", trainBlockSize, 0.5, 0, 0, 0);
	printf("\nLoaded network\n");
	int correctAmountTestData = testNetworkByEvalData(*loadedNet, inputsTest, outputsTest, mnistTestImages.count);
	printf("\nCorrect test data: %d\n", correctAmountTestData);
	destroyNetwork(&loadedNet);
*/
	free(inputs);
	free(outputs);
	free(inputsTest);
	free(outputsTest);
	destroyNetwork(&nn);
	freeMNIST(mnistTrainImages);
	freeMNIST(mnistTrainLabels);
	freeMNIST(mnistTestImages);
	freeMNIST(mnistTestLabels);
}

int main() {
	setupRandom();
	//setupLogs(false);

	startThreading();

	//testXOR();
	//testOR();
	//testAND();

	testMNIST();
	//testAutoencoderMNIST();

	stopThreading();

	//closeLogs();

	// Wait for all subthreads to finish, to prevent valgrind from false leaks, but turns out it still gives some false positives on linux.
	//pthread_exit(NULL);
	return 0;
}
