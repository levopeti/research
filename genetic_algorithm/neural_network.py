import numpy as np
import tensorflow as tf
import time
import pickle
import os
from abc import ABC, abstractmethod
from scipy.signal import convolve2d
from skimage.measure import block_reduce

np.random.seed(int(time.time()))


class NeuralNet(object):
    def __init__(self, input, output):
        self.learning_rate = 0.1
        self.loss = None
        self.error = None

        self.X = None
        self.Y = None
        self.Y_am = None        # Y argmax

        self.batch_size = None
        self.num_batches = None

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

        for b in range(self.num_batches):
            print(b)

            # forward process
            start, end = b * self.batch_size, (b + 1) * self.batch_size
            self.forward(start, end)
            o = self.output.output

            loss, predicted_value = self.loss(o, self.Y[start: end])
            predicted_values.append(predicted_value)

            global_loss += loss

        predicted_values = np.array(predicted_values).reshape(-1,)

        return global_loss, self.accurate_func(np.array(predicted_values))

    def train_step(self):
        """Train one epoch on the network with backpropagation."""

        for b in range(self.num_batches):
            print(b)

            # forward
            start_time = time.time()
            start, end = b * self.batch_size, (b + 1) * self.batch_size
            self.forward(start, end)
            o = self.output.output
            # print("Time of forward: {}s".format(time.time() - start_time))
            error = self.error(o, self.Y[start: end])

            # backward
            start_time = time.time()
            self.backward(error)
            # print("Time of backward: {}s".format(time.time() - start_time))
            # input()

    def forward(self, start, end):
        self.input.forward_process(self.X[start: end])

    def backward(self, error):
        # for the first layer the output_bp = error
        self.output.backward_process(error)

    def train(self, X, Y, epochs, batch_size):
        self.X = X
        self.Y = Y
        self.Y_am = np.argmax(Y, axis=1)

        self.batch_size = batch_size
        self.num_batches = self.X.shape[0] // self.batch_size

        print("Start training the model...\n")

        for i in range(epochs):
            start = time.time()
            self.train_step()
            loss_value, accurate = self.evaluate()

            print("EPOCH", i + 1, "\tAccurate: {0:.2f}%\t".format(accurate * 100), "Loss: {0:.4f}\t".format(loss_value), "ETA: {0:.2f}s\n".format(time.time() - start))
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
            return np.square(o - y).sum() * 0.5 / self.X.shape[0], np.argmax(o, axis=1)

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
            instance.__init__(units, activation=activation, learning_rate=learning_rate, prev_layer=layer)
            return instance
        return set_prev_layer

    def __init__(self, units, activation="sigmoid", learning_rate=0.1, prev_layer=None):
        super().__init__(activation=activation, learning_rate=learning_rate, prev_layer=prev_layer)
        self.batch_size = 100
        self.output_size = (self.batch_size, units)

        self.W = (np.random.rand(self.input_size[1], self.output_size[1]) * 1) - 0.5
        self.b = (np.random.rand(self.output_size[1]) * 1) - 0.5
        # self.b = np.zeros(self.output_size)

        log = "Dense layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(self.W.size + self.b.size, self.input_size, self.output_size)
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
        # TODO: More next layer
        d_act_z = self.d_act(self.z)

        delta_b = np.multiply(input_error, d_act_z)
        x = self.prev_layer.output
        x = np.transpose(x)

        self.output_bp = np.dot(delta_b, np.transpose(self.W))

        delta_W = np.dot(x, delta_b)
        delta_b = np.sum(delta_b, axis=0)

        self.W = np.subtract(self.W, delta_W * (self.learning_rate / self.batch_size))
        self.b = np.subtract(self.b, delta_b * (self.learning_rate / self.batch_size))

        self.prev_layer.backward_process(self.output_bp)


class Conv2d(Layer):
    def __new__(cls, number_of_kernel, kernel_size, activation="sigmoid", learning_rate=0.1):
        def set_prev_layer(layer):
            instance = super(Conv2d, cls).__new__(cls)
            instance.__init__(number_of_kernel, kernel_size, activation=activation, learning_rate=learning_rate, prev_layer=layer)
            return instance
        return set_prev_layer

    def __init__(self, number_of_kernel, kernel_size, activation="sigmoid", learning_rate=0.1, prev_layer=None):
        super().__init__(activation=activation, learning_rate=learning_rate, prev_layer=prev_layer)
        self.kernel_size = kernel_size
        self.number_of_kernel = number_of_kernel
        self.batch_size = 100

        # with 'valid' convolution
        self.output_size = (self.number_of_kernel * self.input_size[0], self.input_size[1] - (self.kernel_size - 1), self.input_size[2] - (self.kernel_size - 1))

        self.W = (np.random.rand(self.number_of_kernel, self.kernel_size, self.kernel_size) * 1) - 0.5
        self.b = (np.random.rand(self.number_of_kernel) * 1) - 0.5
        # self.b = np.zeros(self.output_size)

        log = "2D convolution layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(self.W.size + self.b.size, self.input_size, self.output_size)
        print(log)

    def forward_process(self):
        """2d convolution layer forward process."""
        batch_input = self.prev_layer.output
        self.z = []

        for conv_inputs in batch_input:
            batch = []
            for conv_input in conv_inputs:
                for i in range(self.W.shape[0]):
                    tmp = convolve2d(conv_input, self.W[i], mode="valid") + self.b[i]
                    batch.append(tmp)
            self.z.append(batch)

        self.z = np.array(self.z)
        self.output = self.act(self.z)

        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """2d convolution layer backward process"""
        g = input_error
        d_act_z = self.d_act(self.z)
        delta_batch = g * d_act_z

        output_bp = []
        for b, delta in enumerate(delta_batch):
            batch = []
            tmp_bp = np.zeros((delta[0].shape[0] + (self.W[0].shape[0] - 1), delta[0].shape[1] + (self.W[0].shape[1] - 1)))
            tmp_d = np.zeros((self.W.shape[0], delta[0].shape[0], delta[0].shape[1]))

            j = 0
            for i in range(delta.shape[0]):
                tmp_bp += convolve2d(delta[i], self.W[j], mode="full")
                tmp_d[j] += delta[i]

                j += 1
                if j == self.W.shape[0]:
                    batch.append(tmp_bp)
                    tmp_bp = np.zeros((delta[0].shape[0] + (self.W[0].shape[0] - 1), delta[0].shape[1] + (self.W[0].shape[1] - 1)))
                    j = 0
            output_bp.append(batch)

            tmp_d /= delta.shape[0] / self.W.shape[0]
            # TODO: accelerate and delta_b
            for i in range(tmp_d.shape[0]):
                avg = np.average(tmp_d[i].reshape(1, -1))
                self.b[i] = np.subtract(self.b[i], avg * self.learning_rate)

            for i in range(self.W.shape[0]):
                # delta_W for every kernel (W[i]) with g[i] * dL[i] (delta[i])
                delta_W = np.zeros(self.W[0].shape)

                for x in self.prev_layer.output[b]:
                    delta_W += convolve2d(x, delta[i], mode="valid")

                self.W[i] = np.subtract(self.W[i], delta_W * self.learning_rate)

        self.output_bp = np.array(output_bp)
        self.prev_layer.backward_process(self.output_bp)


class Flatten(Layer):
    def __new__(cls):
        def set_prev_layer(layer):
            instance = super(Flatten, cls).__new__(cls)
            instance.__init__(prev_layer=layer)
            return instance
        return set_prev_layer

    def __init__(self, prev_layer=None):
        super().__init__(prev_layer=prev_layer)
        self.batch_size = 100
        self.set_output_size()

        log = "Flatten layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(0, self.input_size, self.output_size)
        print(log)

    def set_output_size(self):
        output_size = 1
        for i in self.input_size:
            output_size *= i
        self.output_size = (1, output_size)

    def forward_process(self):
        """Flatten connected layer forward process."""
        x = self.prev_layer.output
        self.output = x.reshape(self.batch_size, -1)

        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """Flatten connected layer backward process"""
        # TODO: More next layer
        self.output_bp = input_error.reshape((self.batch_size, self.prev_layer.output_size[0], self.prev_layer.output_size[1], self.prev_layer.output_size[2]))
        self.prev_layer.backward_process(self.output_bp)


class Pool2d(Layer):
    def __new__(cls, kernel_size):
        def set_prev_layer(layer):
            instance = super(Pool2d, cls).__new__(cls)
            instance.__init__(prev_layer=layer, kernel_size=kernel_size)
            return instance
        return set_prev_layer

    def __init__(self, kernel_size, prev_layer=None):
        super().__init__(prev_layer=prev_layer)
        self.kernel_size = kernel_size
        self.batch_size = 100
        self.set_output_size()

        log = "2D pool layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(0, self.input_size, self.output_size)
        print(log)

    def set_output_size(self):
        output = block_reduce(np.zeros((self.input_size[1], self.input_size[2])), block_size=(self.kernel_size, self.kernel_size), func=np.max)
        self.output_size = (self.input_size[0], output.shape[0], output.shape[1])

    def forward_process(self):
        """Flatten connected layer forward process."""
        output = []
        batch_input = self.prev_layer.output

        for pool_input in batch_input:
            batch = []
            for x in pool_input:
                batch.append(block_reduce(x, block_size=(self.kernel_size, self.kernel_size), func=np.max))
            output.append(batch)

        self.output = np.array(output)

        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """Flatten connected layer backward process"""
        # TODO: More next layer
        output = []

        for j in range(self.batch_size):
            batch = []
            for i in range(self.prev_layer.output_size[0]):
                tmp = input_error[j][i].repeat(self.kernel_size, axis=0).repeat(self.kernel_size, axis=1)
                tmp = tmp[:self.prev_layer.output_size[1], :self.prev_layer.output_size[2]]

                tmp2 = self.output[j][i].repeat(self.kernel_size, axis=0).repeat(self.kernel_size, axis=1)
                tmp2 = tmp2[:self.prev_layer.output_size[1], :self.prev_layer.output_size[2]]

                mask = np.equal(self.prev_layer.output[j][i], tmp2).astype(int)
                tmp = tmp * mask
                batch.append(tmp)
            output.append(batch)

        self.output_bp = np.array(output)
        self.prev_layer.backward_process(self.output_bp)


# TODO: class Concat:
# TODO: class Add:
# TODO: class Subtract:
# TODO: class Dropout:
# TODO: class Batchnorm:
# TODO: class sparse XE:


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
    # x_train = np.reshape(x_train, (-1, 28 * 28))
    # x_test = np.reshape(x_test, (-1, 28 * 28))
    x_train = np.reshape(x_train, (-1, 1, 28, 28))
    x_test = np.reshape(x_test, (-1, 1, 28, 28))

    X = np.array(np.append(x_train, x_test, axis=0))
    Y = np.eye(num_class)[np.append(y_train, y_test)]  # one hot vectors

    ip = Input(input_size=(1, 28, 28))
    x = Conv2d(number_of_kernel=10, kernel_size=5, activation="relu")(ip)
    x = Pool2d(kernel_size=3)(x)
    x = Conv2d(number_of_kernel=10, kernel_size=3, activation="relu")(x)
    x = Pool2d(kernel_size=2)(x)
    x = Flatten()(x)
    x = Dense(units=128, activation="sigmoid", learning_rate=1)(x)
    x = Dense(units=64, activation="sigmoid", learning_rate=1)(x)
    op = Dense(units=num_class, activation="sigmoid", learning_rate=1)(x)

    nn = NeuralNet(ip, op)
    nn.build_model(loss="MSE", learning_rate=0.1)
    nn.train(X[:60000], Y[:60000], epochs=5, batch_size=100)
    # nn.save_weights()

    # TODO: optimizers
    # TODO: batch size in layers


