import numpy as np
import time

from abc import ABC, abstractmethod
from scipy.signal import convolve2d
from skimage.measure import block_reduce
from concurrent.futures import ThreadPoolExecutor
# from multiprocessing import cpu_count

np.random.seed(int(time.time()))

POOL = False
MAX_WORKERS = 2


class Input(object):
    def __init__(self, input_size: tuple):
        self.input_size = input_size
        self.output_size = None
        self.batch_size = None

        self.output = None
        self.next_layer = []

    def set_size_forward(self, batch_size, learning_rate, optimizer):
        self.batch_size = batch_size
        output_size = [batch_size]

        for size in self.input_size:
            output_size.append(size)

        self.output_size = tuple(output_size)

        for layer in self.next_layer:
            layer.set_size_forward(batch_size, learning_rate, optimizer)

    def set_next_layer(self, layer):
        self.next_layer.append(layer)

    def forward_process(self, x):
        self.output = x
        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        # print("input backward")
        # input()
        pass

    def save_weights(self, w_array):
        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        for layer in self.next_layer:
            layer.load_weights(w_array)


class Layer(ABC):
    def __init__(self, activation="sigmoid", learning_rate=None, prev_layer=None):
        self.input_size = None
        self.output_size = None
        self.batch_size = None
        self.learning_rate = learning_rate

        self.act = self.act_func(activation)
        self.d_act = self.d_act_func(activation)

        self.output = None
        self.output_bp = None

        self.W = None
        self.b = None
        self.z = None
        self.z_nesterov = None

        # cache for optimizer
        self.cache_W = None
        self.cache_b = None

        self.prev_layer = prev_layer
        self.next_layer = []

        self.prev_layer_set_next_layer()

        self.optimizer = None

    @abstractmethod
    def set_size_forward(self, batch_size, learning_rate, optimizer):
        pass

    @abstractmethod
    def forward_process(self):
        pass

    @abstractmethod
    def backward_process(self, input_error):
        pass

    @abstractmethod
    def save_weights(self, w_array):
        pass

    @abstractmethod
    def load_weights(self, w_array):
        pass

    def set_next_layer(self, layer):
        self.next_layer.append(layer)

    def prev_layer_set_next_layer(self):
        self.prev_layer.set_next_layer(self)

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

    def update_weights(self, delta_W, delta_b):
        """Update weights and velocity of the weights."""
        self.W, self.b, self.cache_W, self.cache_b = self.optimizer.run(self.W, self.cache_W, delta_W, self.b, self.cache_b, delta_b, self.learning_rate)


class Dense(Layer):
    def __new__(cls, units, activation="sigmoid", learning_rate=None):
        def set_prev_layer(layer):
            instance = super(Dense, cls).__new__(cls)
            instance.__init__(units, activation=activation, learning_rate=learning_rate, prev_layer=layer)
            return instance
        return set_prev_layer

    def __init__(self, units, activation="sigmoid", learning_rate=None, prev_layer=None):
        super().__init__(activation=activation, learning_rate=learning_rate, prev_layer=prev_layer)
        self.units = units

    def set_size_forward(self, batch_size, learning_rate, optimizer):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.input_size = self.prev_layer.output_size
        self.output_size = (self.batch_size, 1, self.units)

        if self.learning_rate is None:
            self.learning_rate = learning_rate

        self.W = (np.random.rand(self.input_size[2], self.output_size[2]) * 1) - 0.5
        self.b = (np.random.rand(self.output_size[2]) * 1) - 0.5

        # self.b = np.zeros(self.output_size)

        # init cache in case nasterov
        self.cache_W, self.cache_b = self.optimizer.init(self.W.shape, self.b.shape)

        log = "Dense layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(self.W.size + self.b.size, self.input_size, self.output_size)
        print(log)

        for layer in self.next_layer:
            layer.set_size_forward(batch_size, learning_rate, optimizer)

    def save_weights(self, w_array):
        w_array.append(self.W)
        w_array.append(self.b)

        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        assert w_array[0].shape == self.W.shape and w_array[1].shape == self.b.shape

        self.W = w_array[0]
        self.b = w_array[1]

        w_array = w_array[2:]

        for layer in self.next_layer:
            layer.load_weights(w_array)

    def forward_process(self):
        """Fully connected layer forward process."""
        x = self.prev_layer.output
        self.z = np.add(np.dot(x, self.W), self.b)

        if self.optimizer.name == "SGD":
            if self.optimizer.nesterov:
                nesterov_W = np.subtract(self.W, self.optimizer.gamma * self.cache_W)
                nesterov_b = np.subtract(self.b, self.optimizer.gamma * self.cache_b)
                self.z_nesterov = np.add(np.dot(x, nesterov_W), nesterov_b)

        self.output = self.act(self.z)

        assert self.output.shape == self.output_size

        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """Fully connected layer backward process"""
        # print("dense backward")

        if self.z_nesterov is not None:
            d_act_z = self.d_act(self.z_nesterov)
        else:
            d_act_z = self.d_act(self.z)

        delta_b = np.multiply(input_error, d_act_z)
        x = self.prev_layer.output

        # transpose the last two axis
        x = np.transpose(x, axes=(0, 2, 1))

        self.output_bp = np.dot(delta_b, np.transpose(self.W))

        delta_W = np.tensordot(x, delta_b, axes=([0, 2], [0, 1]))
        delta_b = np.sum(delta_b, axis=0).reshape(-1)

        # normalization
        delta_W = delta_W / self.batch_size
        delta_b = delta_b / self.batch_size

        self.update_weights(delta_W, delta_b)

        assert self.output_bp.shape == self.input_size

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

        self.W = (np.random.rand(self.number_of_kernel, self.kernel_size, self.kernel_size) * 1) - 0.5
        self.b = (np.random.rand(self.number_of_kernel) * 1) - 0.5
        # self.b = np.zeros(self.output_size)

    def set_size_forward(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.input_size = self.prev_layer.output_size

        if self.learning_rate is None:
            self.learning_rate = learning_rate

        # with 'valid' convolution
        self.output_size = (self.batch_size, self.number_of_kernel * self.input_size[1], self.input_size[2] - (self.kernel_size - 1), self.input_size[3] - (self.kernel_size - 1))

        log = "2D convolution layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(self.W.size + self.b.size, self.input_size, self.output_size)
        print(log)

        for layer in self.next_layer:
            layer.set_size_forward(batch_size, learning_rate)

    def save_weights(self, w_array):
        w_array.append(self.W)
        w_array.append(self.b)

        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        assert w_array[0].shape == self.W.shape and w_array[1].shape == self.b.shape

        self.W = w_array[0]
        self.b = w_array[1]

        w_array = w_array[2:]

        for layer in self.next_layer:
            layer.load_weights(w_array)

    def forward_process(self):
        """2d convolution layer forward process."""
        batch_input = self.prev_layer.output
        self.z = []

        if POOL:
            def conv_on_batch(conv_inputs):
                batch = []
                for conv_input in conv_inputs:
                    for i in range(self.W.shape[0]):
                        tmp = convolve2d(conv_input, self.W[i], mode="valid") + self.b[i]
                        batch.append(tmp)
                return batch

            # p = Pool(cpu_count())

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as p:
                results = p.map(conv_on_batch, batch_input)

            for result in results:
                self.z.append(result)

            # p.terminate()
        else:
            for conv_inputs in batch_input:
                batch = []
                for conv_input in conv_inputs:
                    for i in range(self.W.shape[0]):
                        tmp = convolve2d(conv_input, self.W[i], mode="valid") + self.b[i]
                        batch.append(tmp)
                self.z.append(batch)

        self.z = np.array(self.z)

        self.z = np.array(self.z)
        self.output = self.act(self.z)

        assert self.output.shape == self.output_size

        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """2d convolution layer backward process"""
        # print("conv2d backward")
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
        assert self.output_bp.shape == self.input_size

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

    def set_size_forward(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.input_size = self.prev_layer.output_size

        self.set_output_size()

        log = "Flatten layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(0, self.input_size, self.output_size)
        print(log)

        for layer in self.next_layer:
            layer.set_size_forward(batch_size, learning_rate)

    def save_weights(self, w_array):
        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        for layer in self.next_layer:
            layer.load_weights(w_array)

    def set_output_size(self):
        output_size = 1
        for i in self.input_size[1:]:
            output_size *= i
        self.output_size = (self.batch_size, 1, output_size)

    def forward_process(self):
        """Flatten connected layer forward process."""
        x = self.prev_layer.output
        self.output = x.reshape(self.batch_size, 1, -1)

        assert self.output.shape == self.output_size

        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """Flatten connected layer backward process"""
        # print("flatten backward")
        self.output_bp = input_error.reshape(self.prev_layer.output_size)
        assert self.output_bp.shape == self.input_size

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

    def set_size_forward(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.input_size = self.prev_layer.output_size

        self.set_output_size()

        log = "2D pool layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(0, self.input_size, self.output_size)
        print(log)

        for layer in self.next_layer:
            layer.set_size_forward(batch_size, learning_rate)

    def set_output_size(self):
        output = block_reduce(np.zeros((self.input_size[2], self.input_size[3])), block_size=(self.kernel_size, self.kernel_size), func=np.max)
        self.output_size = (self.batch_size, self.input_size[1], output.shape[0], output.shape[1])

    def save_weights(self, w_array):
        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        for layer in self.next_layer:
            layer.load_weights(w_array)

    def forward_process(self):
        """Flatten connected layer forward process."""
        output = []
        batch_input = self.prev_layer.output

        if POOL:
            def pool_on_batch(pool_input):
                batch = []
                for x in pool_input:
                    batch.append(block_reduce(x, block_size=(self.kernel_size, self.kernel_size), func=np.max))
                return batch

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as p:
                results = p.map(pool_on_batch, batch_input)

            for result in results:
                output.append(result)
        else:
            for pool_input in batch_input:
                batch = []
                for x in pool_input:
                    batch.append(block_reduce(x, block_size=(self.kernel_size, self.kernel_size), func=np.max))
                output.append(batch)

        self.output = np.array(output)

        assert self.output.shape == self.output_size

        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """Flatten connected layer backward process"""
        # print("pool2d backward")
        output = []

        if POOL:
            def pool_bp_on_batch(j):
                batch = []
                for i in range(self.prev_layer.output_size[1]):
                    tmp = input_error[j][i].repeat(self.kernel_size, axis=0).repeat(self.kernel_size, axis=1)
                    tmp = tmp[:self.prev_layer.output_size[2], :self.prev_layer.output_size[3]]

                    tmp2 = self.output[j][i].repeat(self.kernel_size, axis=0).repeat(self.kernel_size, axis=1)
                    tmp2 = tmp2[:self.prev_layer.output_size[2], :self.prev_layer.output_size[3]]

                    mask = np.equal(self.prev_layer.output[j][i], tmp2).astype(int)
                    tmp = tmp * mask
                    batch.append(tmp)
                return batch

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as p:
                results = p.map(pool_bp_on_batch, list(range(self.batch_size)))

            for result in results:
                output.append(result)
        else:
            for j in range(self.batch_size):
                batch = []
                for i in range(self.prev_layer.output_size[1]):
                    input_error[j][i].repeat(self.kernel_size, axis=0).repeat(self.kernel_size, axis=1)
                    tmp = input_error[j][i].repeat(self.kernel_size, axis=0).repeat(self.kernel_size, axis=1)
                    tmp = tmp[:self.prev_layer.output_size[2], :self.prev_layer.output_size[3]]

                    tmp2 = self.output[j][i].repeat(self.kernel_size, axis=0).repeat(self.kernel_size, axis=1)
                    tmp2 = tmp2[:self.prev_layer.output_size[2], :self.prev_layer.output_size[3]]

                    mask = np.equal(self.prev_layer.output[j][i], tmp2).astype(int)
                    tmp = tmp * mask
                    batch.append(tmp)
                output.append(batch)

        self.output_bp = np.array(output)
        assert self.output_bp.shape == self.input_size

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
        self.axis = axis + 1  # because the batch size

    def set_size_forward(self, batch_size, learning_rate):
        os0 = self.prev_layer[0].output_size
        os1 = self.prev_layer[1].output_size

        if os0 is not None and os1 is not None:
            self.batch_size = batch_size

            self.set_output_size()
            self.set_input_size()

            log = "Concat layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(0, self.input_size, self.output_size)
            print(log)

            for layer in self.next_layer:
                layer.set_size_forward(batch_size, learning_rate)
        else:
            # It takes the process back to a not calculated layer.
            pass

    def prev_layer_set_next_layer(self):

        for layer in self.prev_layer:
            layer.set_next_layer(self)

    def set_input_size(self):
        input_size = []

        for layer in self.prev_layer:
            input_size.append(layer.output_size)

        self.input_size = tuple(input_size)

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

    def save_weights(self, w_array):
        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        for layer in self.next_layer:
            layer.load_weights(w_array)

    def forward_process(self):
        """Concatenate layer forward process."""
        # TODO: by axis = 0 order alternately
        x0 = self.prev_layer[0].output
        x1 = self.prev_layer[1].output

        if x0 is not None and x1 is not None:
            try:
                self.output = np.concatenate((x0, x1), axis=self.axis)
            except ValueError:
                print("In layer Concat, the two layers don't have the same dimension at the appropriate axis!")

            assert self.output.shape == self.output_size

            for layer in self.next_layer:
                layer.forward_process()
        else:
            # It takes the process back to a not calculated layer.
            pass

    def backward_process(self, input_error):
        """Concatenate layer backward process"""
        # print("concat backward")
        # TODO: by axis = 0 order alternately
        s0 = self.prev_layer[0].output_size[self.axis]
        s1 = self.prev_layer[1].output_size[self.axis]

        self.output_bp = np.split(input_error, [s0], axis=self.axis)

        assert self.output_bp[1].shape[self.axis] == s1

        for i, layer in enumerate(self.prev_layer):
            layer.backward_process(self.output_bp[i])


class Add(Layer):
    def __new__(cls, weights_of_layers=None):
        def set_prev_layer(layer):
            """
            layer: list of the added layers [x1, x2]
            """
            instance = super(Add, cls).__new__(cls)
            instance.__init__(prev_layer=layer, weights_of_layers=None)
            return instance
        return set_prev_layer

    def __init__(self, prev_layer=None, weights_of_layers=None):
        super().__init__(prev_layer=prev_layer)
        # weigths at the addition
        if weights_of_layers:
            self.weights_of_layers = weights_of_layers
        else:
            self.weights_of_layers = [1, 1]

    def set_size_forward(self, batch_size, learning_rate):
        os0 = self.prev_layer[0].output_size
        os1 = self.prev_layer[1].output_size

        if os0 is not None and os1 is not None:
            assert os0 == os1
            self.batch_size = batch_size

            self.output_size = os0
            self.input_size = os0

            log = "Add layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(0, self.input_size, self.output_size)
            print(log)

            for layer in self.next_layer:
                layer.set_size_forward(batch_size, learning_rate)
        else:
            # It takes the process back to a not calculated layer.
            pass

    def prev_layer_set_next_layer(self):

        for layer in self.prev_layer:
            layer.set_next_layer(self)

    def save_weights(self, w_array):
        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        for layer in self.next_layer:
            layer.load_weights(w_array)

    def forward_process(self):
        """Concatenate layer forward process."""
        # TODO: by axis = 0 order alternately
        x0 = self.prev_layer[0].output
        x1 = self.prev_layer[1].output

        if x0 is not None and x1 is not None:
            try:
                x0 *= self.weights_of_layers[0]
                x1 *= self.weights_of_layers[1]

                self.output = np.add(x0, x1)
            except ValueError:
                print("In layer Add, the two layers don't have the same shape!")

            assert self.output.shape == self.output_size

            for layer in self.next_layer:
                layer.forward_process()
        else:
            # It takes the process back to a not calculated layer.
            pass

    def backward_process(self, input_error):
        """Concatenate layer backward process"""
        # print("add backward")
        self.output_bp = input_error
        assert self.output_bp.shape == self.input_size

        for i, layer in enumerate(self.prev_layer):
            layer.backward_process(self.output_bp * self.weights_of_layers[i])

# TODO: class Dropout:
# TODO: class Batchnorm:
# TODO: init weights


