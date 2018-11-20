import numpy as np
import tensorflow as tf
import time
import pickle
import os
from abc import ABC, abstractmethod
from scipy.signal import convolve2d
from skimage.measure import block_reduce
from pathos.multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

np.random.seed(int(time.time()))


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
            instance.__init__(number_of_kernel=number_of_kernel, kernel_size=kernel_size, activation=activation, learning_rate=learning_rate, prev_layer=layer)
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

        def conv_on_batch(conv_inputs):
            batch = []
            for conv_input in conv_inputs:
                for i in range(self.W.shape[0]):
                    tmp = convolve2d(conv_input, self.W[i], mode="valid") + self.b[i]
                    batch.append(tmp)
            return batch

        # p = Pool(cpu_count())

        with ThreadPoolExecutor(max_workers=36) as p:
            results = p.map(conv_on_batch, batch_input)

        for result in results:
            self.z.append(result)

        # p.terminate()
        self.z = np.array(self.z)

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

        def pool_on_batch(pool_input):
            batch = []
            for x in pool_input:
                batch.append(block_reduce(x, block_size=(self.kernel_size, self.kernel_size), func=np.max))
            return batch

        with ThreadPoolExecutor(max_workers=36) as p:
            results = p.map(pool_on_batch, batch_input)

        for result in results:
            output.append(result)

        self.output = np.array(output)

        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """Flatten connected layer backward process"""
        # TODO: More next layer
        output = []

        def pool_bp_on_batch(j):
            batch = []
            for i in range(self.prev_layer.output_size[0]):
                tmp = input_error[j][i].repeat(self.kernel_size, axis=0).repeat(self.kernel_size, axis=1)
                tmp = tmp[:self.prev_layer.output_size[1], :self.prev_layer.output_size[2]]

                tmp2 = self.output[j][i].repeat(self.kernel_size, axis=0).repeat(self.kernel_size, axis=1)
                tmp2 = tmp2[:self.prev_layer.output_size[1], :self.prev_layer.output_size[2]]

                mask = np.equal(self.prev_layer.output[j][i], tmp2).astype(int)
                tmp = tmp * mask
                batch.append(tmp)
            return batch

        with ThreadPoolExecutor(max_workers=36) as p:
            results = p.map(pool_bp_on_batch, list(range(self.batch_size)))

        for result in results:
            output.append(result)

        self.output_bp = np.array(output)
        self.prev_layer.backward_process(self.output_bp)


class Concat(Layer):
    def __new__(cls, axis=1):
        def set_prev_layer(layer):
            """
            layer: list of the concatenated layers [x1, x2]
            """
            instance = super(Concat, cls).__new__(cls)
            instance.__init__(prev_layer=layer, axis=axis)
            return instance
        return set_prev_layer

    def __init__(self, prev_layer=None, axis=1):
        super().__init__(prev_layer=prev_layer)
        self.axis = axis
        self.batch_size = 100
        self.set_output_size()

        log = "Flatten layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(0, self.input_size, self.output_size)
        print(log)

    def set_prev_layer(self):
        self.input_size = []

        for layer in self.prev_layer:
            layer.set_next_layer(self)
            self.input_size.append(layer.output_size)

    def set_output_size(self):
        # TODO: by axis = 0 order alternately
        output_size_at_axis = 0
        output_size = []

        for layer in self.prev_layer:
            output_size_at_axis += layer.output_size[self.axis]

        for i, size in enumerate(self.prev_layer[0].output_size):
            if i == self.axis:
                output_size.append(output_size_at_axis)
            else:
                output_size.append(size)

        self.output_size = tuple(output_size)

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


# TODO: class Add:
# TODO: class Subtract:
# TODO: class Dropout:
# TODO: class Batchnorm:
# TODO: class sparse XE:


