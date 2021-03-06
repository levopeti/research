import numpy as np
import tensorflow as tf
import time
import pickle
import os
import cudamat as cm
from cudamat import CUDAMatrix as cmarray

cm.cuda_set_device(0)
np.random.seed(int(time.time()))


class NeuralNet(object):
    def __init__(self):
        self.num_class = 10
        self.learning_rate = 1

        loss = "MSE"
        self.loss = self.loss_func(loss)
        self.error = self.error_func(loss)

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        x_train = np.reshape(x_train, (-1, 28 * 28))
        x_test = np.reshape(x_test, (-1, 28 * 28))

        x_train = np.float32(x_train)
        x_test = np.float32(x_test)

        # self.X = np.array(np.append(x_train, x_test, axis=0))
        self.X = np.array(x_train)
        # self.Y = np.eye(self.num_class)[np.append(y_train, y_test)]  # one hot vectors
        self.Y_argmax = y_train
        self.Y = np.eye(self.num_class)[y_train]  # one hot vectors

        self.input_length = len(self.X[0])
        self.train_data_length = len(self.X)

        self.batch_size = 100
        self.num_batches = self.X.shape[0] // self.batch_size

        cm.init()

        self.X = cmarray(self.X)
        self.Y = cmarray(self.Y)
        self.Y_argmax = cmarray(self.Y_argmax)

        self.error_values = cm.empty((self.batch_size, self.num_class))

        self.model = []
        self.build_model()

    def __del__(self):
        del self.X
        del self.Y
        del self.error_values
        # cm.shutdown()

    def build_model(self):
        print("Build the model...\n")
        self.model.append(Layer("fc", self.input_length, 10, self.learning_rate, "sigmoid", self.batch_size))
        # self.model.append(Layer("fc", 128, 64, self.learning_rate, "sigmoid", self.batch_size))
        # self.model.append(Layer("fc", 64, 10, self.learning_rate, "sigmoid", self.batch_size))

    def set_weights(self, individual):
        self.W = np.reshape(np.array(individual[:7840]), (784, 10))  # shape (784, 10)
        self.b = np.array(individual[-10:])  # shape (10,)

    def get_weights_as_genes(self):
        return np.concatenate((np.reshape(self.W, (7840,)), self.b), axis=None)

    def save_weights(self):
        weights = []
        for layer in self.model:
            weights.append(layer.W.asnumpy())
            weights.append(layer.b.asnumpy())

        with open(os.path.join('weights.txt'), 'wb') as fp:
            pickle.dump(weights, fp)

    def evaluate(self):
        """Evaluate the model."""
        global_loss = 0
        predicted_values = []

        for b in range(self.num_batches):

            # forward
            start, end = b * self.batch_size, (b + 1) * self.batch_size
            o = self.forward(start, end)

            loss, predicted_value = self.loss(o, self.Y.get_row_slice(start, end))

            global_loss += loss
            predicted_values.append(predicted_value)

        predicted_values = np.array(predicted_values).reshape(-1,)

        return global_loss, self.accurate_func(predicted_values)

    def train_step(self):
        """Train one epoch on the network with backpropagation."""

        # alternative async

        for b in range(self.num_batches):
            # print(b)

            # forward
            start_time = time.time()
            start, end = b * self.batch_size, (b + 1) * self.batch_size
            o = self.forward(start, end)
            # print("Time of forward: {}s".format(time.time() - start_time))
            self.error(o, self.Y.get_row_slice(start, end))

            # backward
            start_time = time.time()
            self.backward(start, end)
            # print("Time of backward: {}s".format(time.time() - start_time))
            # input()

    def forward(self, start, end):
        data = self.X.get_row_slice(start, end)
        for layer in self.model:
            data = layer.forward(data)
        return data

    def backward(self, start, end):
        output_bp = self.error_values
        for j in range(len(self.model))[::-1]:
            layer = self.model[j]
            if j == 0:
                output_bp = layer.backward(self.X.get_row_slice(start, end), output_bp)
            else:
                output_bp = layer.backward(self.model[j - 1].output, output_bp)

            # if j == len(self.model) - 1 and len(self.model) != 1:
            #     output_bp = layer.backward(self.model[j - 1].output, output_bp)
            # elif j == 0:
            #     output_bp = layer.backward(self.X[i], output_bp)
            # else:
            #     output_bp = layer.backward(self.model[j - 1].output, output_bp)

    def base_line(self, epochs):
        print("Start training the model...\n")

        for i in range(epochs):
            start = time.time()
            self.train_step()
            loss_value, accurate = self.evaluate()

            print("EPOCH", i + 1, "\tAccurate: {0:.2f}%\t".format(accurate * 100), "Loss: {0:.4f}\t".format(loss_value[0, 0]), "Time: {0:.2f}s\n".format(time.time() - start))
            if i == 20:
                self.learning_rate *= 0.5

    def accurate_func(self, pred):
        goal = 0
        for i in range(pred.shape[0]):
            if pred[i] == self.Y_argmax[i]:
                goal += 1
        return goal / (pred.shape[0])

    def loss_func(self, type):
        def mse(o, y):
            tmp_o = cm.empty(o.shape)
            o.subtract(y, target=tmp_o)
            tmp_o.mult(tmp_o)

            tmp = tmp_o.sum(axis=0)
            tmp = tmp.sum(axis=1)
            tmp.mult(0.5 / self.X.shape[0])

            o.copy_to_host()
            tmp.copy_to_host()
            loss = tmp.numpy_array
            o_cpu = o.numpy_array

            return loss, np.argmax(o_cpu, axis=1)

        def xe(o, y):
            return self.cross_entropy(o, y), nd.argmax(self.softmax(o), axis=1)

        if type == "MSE":
            return mse

        elif type == "XE":
            return xe

    def error_func(self, type):
        def mse(o, y):
            o.subtract(y, target=self.error_values)

        def xe(o, y):
            return self.d_cross_entropy(o, y)

        if type == "MSE":
            return mse

        elif type == "XE":
            return xe

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = nd.exp(x)
        result = nd.divide(nd.transpose(e_x), nd.sum(e_x, axis=1))

        return nd.transpose(result)

    def cross_entropy(self, o, y):
        """
        o is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        print(o[0])
        p = self.softmax(o)
        print(p[0])
        print(y[0])
        k = nd.multiply(y, p)
        print(k[0])
        input()
        # We use multidimensional array indexing to extract
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.

        log_likelihood = -nd.log(nd.max(k, axis=1))
        print(log_likelihood)
        loss = nd.sum(log_likelihood) / m
        print(666, loss)
        return loss

    def d_cross_entropy(self, o, y):
        """
        o is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        y = y.argmax(axis=1)
        m = y.shape[0]
        grad = self.softmax(o)
        grad[:m, y] -= 1
        # input()
        grad = grad / m

        return grad


class Layer(object):
    def __init__(self, layer_type, input_size, output_size, learning_rate, activation, batch_size):
        self.layer_type = layer_type
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.act = self.act_func(activation)
        self.d_act = self.d_act_func(activation)

        self.output = None

        self.W = None
        self.b = None
        self.z = None

        self.forward = None
        self.backward = None

        if self.layer_type == "fc":
            W = np.random.rand(self.input_size, self.output_size) * 1 - 0.5
            b = np.random.rand(1, self.output_size) * 1 - 0.5
            # self.delta_W = np.zeros((self.input_size, self.output_size))
            # self.delta_b = np.zeros(self.output_size)

            self.forward = self.dense_fw
            self.backward = self.dense_bw

            self.W = cmarray(W)
            self.b = cmarray(b)
            self.b_prev = cm.empty(self.b.shape)
            self.delta_W = cm.empty(self.W.shape)
            self.delta_b = cm.empty((self.batch_size, self.output_size))
            self.z = cm.empty((self.batch_size, self.output_size))
            self.d_act_z = cm.empty(self.z.shape)

            self.output = cm.empty(self.z.shape)
            self.tmp1 = cm.empty(self.z.shape)
            self.tmp2 = cm.empty(self.z.shape)

    def __del__(self):
        del self.W
        del self.b
        del self.b_prev
        del self.delta_W
        del self.delta_b
        del self.z
        del self.d_act_z
        del self.output
        del self.tmp1
        del self.tmp2

    def dense_fw(self, x):
        """Fully connected layer forward process."""
        cm.dot(x, self.W, target=self.z)

        self.z.add_row_vec(self.b)
        self.act(self.z)
        self.output.assign(self.tmp1)

        return self.output

    def dense_bw(self, input_layer, input_error):
        """Fully connected layer backward process"""
        self.d_act(self.z)
        self.d_act_z.assign(self.tmp1)

        input_error.mult(self.d_act_z, target=self.delta_b)
        x = input_layer.transpose()

        cm.dot(x, self.delta_b, target=self.delta_W)

        output_bp = cm.dot(self.delta_b, self.W.transpose())

        self.delta_b.sum(axis=0, target=self.b_prev)
        assert self.batch_size == input_error.shape[0]

        self.W.subtract(self.delta_W.mult(self.learning_rate / self.batch_size))
        self.b.subtract(self.b_prev.mult(self.learning_rate / self.batch_size))

        return output_bp

    # def update_weigths(self, batch_size):
    #     self.delta_W = nd.divide(self.delta_W, batch_size)
    #     self.delta_b = nd.divide(self.delta_b, batch_size)
    #     print(self.delta_W.shape)
    #     print(self.delta_b.shape)
    #     input()
    #
    #     self.W = nd.subtract(self.W, self.delta_W * self.learning_rate)
    #     self.b = nd.subtract(self.b, self.delta_b * self.learning_rate)
    #     print(self.W.shape)
    #     print(self.b.shape)
    #     input()
    #
    #     self.delta_W = mx.nd.zeros(self.W.shape, ctx=ctx)
    #     self.delta_b = mx.nd.zeros(self.b.shape, ctx=ctx)

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

    def tanh(self, x):
        return cm.tanh(x)

    def d_tanh(self, x):
        return 1 - cm.tanh(x) ** 2

    def log(self, x):
        self.tmp1.assign(x)

        self.tmp1.mult(-1)
        cm.exp(self.tmp1, target=self.tmp1)
        self.tmp1.add(1)
        cm.pow(self.tmp1, -1)

    def d_log(self, x):
        self.log(x)
        self.tmp2.assign(self.tmp1)
        self.tmp1.mult(-1)
        self.tmp1.add(1)
        self.tmp1.mult(self.tmp2)

    def relu(self, x):
        return nd.maximum(x, 0)

    def d_relu(self, x):
        return nd.where(x > 0, 1, 0)


if __name__ == "__main__":
    nn = NeuralNet()
    nn.base_line(5)
    # nn.save_weights()






