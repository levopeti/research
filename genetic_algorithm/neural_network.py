import numpy as np
import tensorflow as tf
import time
import pickle
import os
from abc import ABC, abstractmethod
from scipy import signal

np.random.seed(int(time.time()))


class NeuralNet(object):
    def __init__(self, input, output):
        self.learning_rate = 0.1
        self.loss = None
        self.error = None

        self.X = None
        self.Y = None

        self.input = input
        self.output = output

        self.model = None

    def __del__(self):
        pass

    def build_model(self, loss="MSE", learning_rate=0.1):
        print("Build the model...\n")

        self.learning_rate = learning_rate
        self.loss = self.loss_func(loss)
        self.error = self.error_func(loss)

        # def compile(layer):
        #     if layer:
        #         for next_layer in layer.next_layer:
        #             next_layer.learning_rate = self.learning_rate
        #             compile(next_layer)
        #
        # compile(self.input)

    def set_weights(self, individual):
        self.W = np.reshape(np.array(individual[:7840]), (784, 10))  # shape (784, 10)
        self.b = np.array(individual[-10:])  # shape (10,)

    def get_weights_as_genes(self):
        return np.concatenate((np.reshape(self.W, (7840,)), self.b), axis=None)

    def save_weights(self):
        weights = []
        for layer in self.model:
            weights.append(layer.W)
            weights.append(layer.b)

        with open(os.path.join('weights.txt'), 'wb') as fp:
            pickle.dump(weights, fp)

    def evaluate(self):
        """Evaluate the model."""
        global_loss = 0
        predicted_values = []

        for i in range(len(self.X)):

            # forward process
            self.forward(i)
            o = self.output.output

            loss, predicted_value = self.loss(o, self.Y[i])
            predicted_values.append(predicted_value)

            global_loss += loss

        return global_loss, self.accurate_func(np.array(predicted_values))

    def train_step(self):
        """Train one epoch on the network with backpropagation."""

        # alternative async

        for i in range(len(self.X)):

            # forward process
            start = time.time()
            self.forward(i)
            o = self.output.output
            # print("Time of forward: {}s\n".format(time.time() - start))
            error = self.error(o, self.Y[i])

            # backward process
            start = time.time()
            self.backward(error, i)
            # print("Time of backward: {}s\n".format(time.time() - start))
            # input()

    def forward(self, i):
        self.input.forward_process(self.X[i])
        # for layer in self.model:
        #     data = layer.forward(data)
        # return data

    def backward(self, output_bp, i):
        # for the first layer the output_bp = error
        self.output.backward_process(output_bp)

        # for j in range(len(self.model))[::-1]:
        #     layer = self.model[j]
        #     if j == 0:
        #         output_bp = layer.backward(self.X[i], output_bp)
        #     else:
        #         output_bp = layer.backward(self.model[j - 1].output, output_bp)

    def train(self, X, Y, epochs):
        self.X = X
        self.Y = Y

        print("Start training the model...\n")

        for i in range(epochs):
            start = time.time()
            self.train_step()
            loss_value, accurate = self.evaluate()

            print("EPOCH", i + 1, "\tAccurate: {0:.2f}%\t".format(accurate * 100), "Loss: {0:.2f}\t".format(loss_value), "ETA: {0:.2f}s\n".format(time.time() - start))
            if i == 30:
                self.learning_rate *= 0.5

    def accurate_func(self, pred):
        goal = 0

        for i in range(pred.shape[0]):

            if pred[i] == np.argmax(self.Y[i]):
                goal += 1
        return goal / pred.shape[0]

    def loss_func(self, type):
        def mse(o, y):
            return np.square(o - y).sum() * 0.5, np.argmax(o)

        def xe(o, y):
            return self.cross_entropy(o, y), np.argmax(self.softmax(o))

        if type == "MSE":
            return mse

        elif type == "XE":
            return xe

    def error_func(self, type):
        def mse(o, y):
            return np.subtract(o, y)

        def xe(o, y):
            return self.d_cross_entropy(o, y)[0]

        if type == "MSE":
            return mse

        elif type == "XE":
            return xe

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def cross_entropy(self, x, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        y = np.array(y).reshape((1, -1))
        x = np.array(x).reshape((1, -1))
        y = y.argmax(axis=1)
        m = y.shape[0]
        p = self.softmax(x)
        # We use multidimensional array indexing to extract
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def d_cross_entropy(self, x, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        y = np.array(y).reshape((1, -1))
        x = np.array(x).reshape((1, -1))
        y = y.argmax(axis=1)
        m = y.shape[0]
        grad = self.softmax(x)
        grad[range(m), y] -= 1
        grad = grad / m
        return grad


class Layer(ABC):
    def __init__(self, activation="sigmoid", learning_rate=0.1, prev_layer=None):
        self.input_size = None
        self.output_size = None
        self.learning_rate = learning_rate

        self.act = self.act_func(activation)
        self.d_act = self.d_act_func(activation)

        self.output = None
        self.output_bp = None

        self.W = None
        self.b = None
        self.z = None

        self.prev_layer = prev_layer
        self.next_layer = []

        self.set_prev_layer()

        # convention of input shape

    def set_prev_layer(self):
        self.prev_layer.set_next_layer(self)
        self.input_size = self.prev_layer.output_size

    @abstractmethod
    def forward_process(self):
        pass

    @abstractmethod
    def backward_process(self, input_error):
        pass

    def set_next_layer(self, layer):
        self.next_layer.append(layer)

    def act_func(self, type):
        if type == "tanh":
            return self.tanh

        elif type == "sigmoid":
            return self.log

        elif type == "relu":
            return self.relu

    def d_act_func(self, type):
        if type == "tanh":
            return self.d_tanh

        elif type == "sigmoid":
            return self.d_log

        elif type == "relu":
            return self.d_relu

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def d_tanh(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def log(x):
        return 1 / (1 + np.exp(-1 * x))

    def d_log(self, x):
        return self.log(x) * (1 - self.log(x))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def d_relu(x):
        return np.where(x > 0, 1, 0)


class Dense(Layer):
    def __new__(cls, units, activation="sigmoid", learning_rate=0.1):
        def set_prev_layer(layer):
            instance = super(Dense, cls).__new__(cls)
            instance.__init__(units, activation="sigmoid", learning_rate=learning_rate, prev_layer=layer)
            return instance
        return set_prev_layer

    def __init__(self, units, activation="sigmoid", learning_rate=0.1, prev_layer=None):
        super().__init__(activation, learning_rate, prev_layer)
        self.output_size = (1, units)

        self.W = (np.random.rand(self.input_size[1], self.output_size[1]) * 1) - 0.5
        self.b = (np.random.rand(self.output_size[1]) * 1) - 0.5
        # self.b = np.zeros(self.output_size)

        log = "Dense layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format((self.input_size[1] + 1) * self.output_size[1], self.input_size[1], self.output_size[1])
        print(log)

    def forward_process(self):
        """Fully connected layer forward process."""
        x = self.prev_layer.output
        self.z = np.add(np.dot(x, self.W), self.b)
        self.output = self.act(self.z)
        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """Fully connected layer backward process"""
        d_act_z = self.d_act(self.z)

        delta_b = np.multiply(input_error, d_act_z)
        delta_b = delta_b.reshape((1, -1))
        x = self.prev_layer.output
        x = np.transpose(x)
        delta_W = np.dot(x, delta_b)

        self.output_bp = np.dot(delta_b, np.transpose(self.W))
        self.W = np.subtract(self.W, delta_W * self.learning_rate)
        self.b = np.subtract(self.b, delta_b * self.learning_rate)

        self.prev_layer.backward_process(self.output_bp)

# class Conv2d(Layer):
#     def __init__(self, input_size, output_size, learning_rate, activation, kernel_size, number_of_kernel):
#         super().__init__(input_size, output_size, learning_rate, activation)
#         self.kernel_size = kernel_size
#         self.number_of_kernel = number_of_kernel
#
#         self.W = (np.random.rand(self.number_of_kernel, self.kernel_size, self.kernel_size) * 1) - 0.5
#         self.b = (np.random.rand(self.number_of_kernel) * 1) - 0.5
#         # self.b = np.zeros(self.output_size)
#         self.forward = self.forward_process
#         self.backward = self.backward_process
#
#     def forward_process(self, x):
#         """2d convolution layer forward process."""
#         # TODO: Bias
#         conv_input = x
#         self.z = []
#
#         for kernel in self.W:
#             self.z.append(signal.convolve2d(conv_input, kernel, mode="same"))
#         self.z = np.array(self.z)
#         self.output = self.act(self.z)
#
#         return self.output
#
#     def backward_process(self, input_layer, input_error):
#         """2d convolution layer backward process"""
#         # TODO: Bias
#         d_act_z = self.d_act(self.z)
#
#         for kernel in self.W:
#
#
# class Concat:
#     pass


class Input(object):
    def __init__(self, input_size: tuple):
        self.output_size = input_size

        self.output = None
        self.next_layer = []

    def set_next_layer(self, layer):
        self.next_layer.append(layer)

    def forward_process(self, x):
        self.output = x
        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        pass


if __name__ == "__main__":
    num_class = 10
    mnist = tf.keras.datasets.mnist
    fashion = False
    if fashion:
        from keras.datasets import fashion_mnist

        mnist = fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.reshape(x_train, (-1, 1, 28 * 28))
    x_test = np.reshape(x_test, (-1, 1, 28 * 28))

    X = np.array(np.append(x_train, x_test, axis=0))
    Y = np.eye(num_class)[np.append(y_train, y_test)]  # one hot vectors

    ip = Input(input_size=(1, 784))
    x = Dense(units=512, activation="sigmoid")(ip)
    x = Dense(units=128, activation="sigmoid")(x)
    op = Dense(units=10, activation="sigmoid")(x)

    nn = NeuralNet(ip, op)
    nn.build_model(loss="MSE", learning_rate=0.1)
    nn.train(X, Y, epochs=60)
    nn.save_weights()


