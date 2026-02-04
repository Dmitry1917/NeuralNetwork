struct MNIST_Data {
	int dimensionsAmount;
	int *dimensions;
	int count;
	unsigned char *data;
};

void freeMNIST(struct MNIST_Data mnist);

struct MNIST_Data readMNIST(char *fileName);
