## Neural network.
This code show examples of simple neural network training and its usage. The training system and calculations are written in C for better performance. There are not proper error handling or multiple checks to prevent wrong usage. Correct memory management was verified by Valgrind.

Network itself is in files **nn_NeuralNetwork.h** **(look for comments, explaining functions)** and nn_NeuralNetwork.c.

File **use_of_neural_network.c** contains examples of network usage for XOR, OR, AND calculations, using different methods, and digits recognition (MNIST database). All of network abilities can be seen there either directly working or in commented sections (like saving/loading or logging of training).

File **test_net.py** do MNIST recognition by using neural network library from Python. It has several versions of training with different parameters and draw results graphs at the end **(commented at the end of file - all together work quite long)**. Also test results are saved in json and can be used for futher analysis to look up for best network configuration **(function processJson())**.

Both files also contain very primitive examples of autoencoder usage.

Files MNIST.h and MNIST.c has functions for reading MNIST db (used in C examples, mentioned above).

In its current state neural network can be created with any amount of layers and neurons as long as memory allows.

Supported neuron activation functions:
- sigmoid;
- tanh;
- ReLU;
- linear;
- softmax.

Softmax can be used only in output layer and together with log likehood cost function.

Supported cost functions:
- square;
- cross entropy;
- log likehood.

Cross entropy need output layer to be sigmoid, and log likehood - softmax.

Network learning can be done by:
- simple gradient descent;
- batch gradient descent;
- mini batch stochastic gradient descent.

The last one is considered main option and supports additional optimizations, such as changing learning rate if network don't improve for too long.

Common optional optimizations:
- L1 regularization;
- L2 regularization;
- weight decay (separated from L2, like in AdamW optimizer in PyTorch, but unlike it, applied at the final stage of weight update, not the first one);
- weights momentum;
- Adam optimizer.

Network feedforward can be executed in 4 threads for performance **(see startThreading/stopThreading function)**.
