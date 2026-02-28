## Neural network.
This code show examples of simple neural network training and its usage. The training system and calculations are written in C for better performance. There are not serious error handling or multiple checks to prevent wrong usage. Proper memory management was tested by Valgrind.

Network itself is in files nn_NeuralNetwork.h **(see for comments, explaining functions)** and nn_NeuralNetwork.c.

File use_of_neural_network.c contains examples of network usage for XOR, OR, AND calculations, using different methods, and digits recognition (MNIST database). All of network abilities can be seen there either directly working or in commented sections (like saving/loading and training logging).
File test_net.py do MNIST recognition by using neural network library from Python. It has several versions of training with different parameters and draw results graphs at the end **(commented at the end of file - all together work quite long).

Files MNIST.h and MNIST.c has functions for reading MNIST db.

In its current state neural network can be created with any amount of layers and neurons as long as memory allow.

Supported neuron activation functions:
- sigmoid;
- tanhi;
- ReLU;
- softmax.
Softmax can be used only in output layer and together with log likehood cost function.

Supported cost functions:
- square;
- cross entropy;
- log likehood.
Cross entropy need output layer to be sigmoid, and log likehood - softmax.

Network learning technics:
- simple gradient descent;
- batch gradient descent;
- mini batch stochastic gradient descent.
The last one is considered main option and supports additional optimizations, such as changing learning rate if network dont improve for too long.

Common optional optimizations:
- L2 regularization;
- weights momentum.

Network feedforward can be executed in 4 threads for performance **(see startThreading/stopThreading function)**.
