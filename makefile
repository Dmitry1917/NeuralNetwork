TARGET = neuralNetwork

CFLAGS = -O3 -Wall -Wextra #-g -fsanitize=address,undefined,leak#,thread

$(TARGET): use_of_neural_network.c mnist.a nn_net.a
	gcc $(CFLAGS) -o $@ $^ -lm

mnist.a: mnist.o
	ar rcs $@ $^

mnist.o: MNIST.c
	gcc $(CFLAGS) -c -o $@ $^

nn_net.a: nn_net.o
	ar rcs $@ $^

nn_net.o: nn_NeuralNetwork.c
	gcc $(CFLAGS) -c -o $@ $^

dynamic: nn_NeuralNetwork.c
	gcc -O3 -fPIC -shared -o nn_net.so $^

clean:
	rm -f *.o *.a $(TARGET)

cleanAll:
	rm -f *.o *.a *.so $(TARGET)
