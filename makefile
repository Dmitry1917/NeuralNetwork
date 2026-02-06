TARGET = neuralNetwork

$(TARGET): use_of_neural_network.c mnist.a nn_net.a
	gcc -o $@ $^ -lm -g

mnist.a: mnist.o
	ar rcs $@ $^

mnist.o: MNIST.c
	gcc -c -o $@ $^

nn_net.a: nn_net.o
	ar rcs $@ $^

nn_net.o: nn_NeuralNetwork.c
	gcc -c -o $@ $^

clean:
	rm -f *.o *.a $(TARGET)
