#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "MNIST.h"

void freeMNIST(struct MNIST_Data mnist) {
	free(mnist.dimensions);
	free(mnist.data);
}

struct MNIST_Data readMNIST(char *fileName) {
	const char *errorMessage = "\nMNIST data is corrupted.\n";
	FILE *file = fopen(fileName, "rb");
	unsigned char mainInfoBuffer[4];
	int *dimensions = NULL;
	int samplesCount;
	size_t readed = fread(mainInfoBuffer, sizeof(mainInfoBuffer), 1, file);
	if(readed < 1) {
		fclose(file);
		fprintf(stderr, "%s", errorMessage);
		exit(1);
	}

	struct MNIST_Data mnist;

	int dimensionsAmount = mainInfoBuffer[3];
	if(dimensionsAmount > 0) {
		int dimensionsBufferSize = dimensionsAmount * sizeof(uint32_t);
		int sampleSize = 1;
		uint32_t *dimensionsBuffer = malloc(dimensionsBufferSize);
		uint32_t *allDimensions = malloc(dimensionsBufferSize);

		readed = fread(dimensionsBuffer, dimensionsBufferSize, 1, file);
		if(readed < 1) {
			fclose(file);
			fprintf(stderr, "%s", errorMessage);
			exit(1);
		}
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
		readed = fread(buf, dataSize, 1, file);
		if(readed < 1) {
			fclose(file);
			fprintf(stderr, "%s", errorMessage);
			exit(1);
		}

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
		fprintf(stderr, "%s", errorMessage);
		exit(1);
	}
	fclose(file);

	return mnist;
}

