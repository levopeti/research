import numpy as np
# import mxnet as mx
# from mxnet import nd
import tensorflow as tf
from keras.datasets import fashion_mnist
import time
import pickle
from pathos.multiprocessing import Pool
from multiprocessing import cpu_count
from scipy import signal
from sklearn.metrics import log_loss
import os

np.random.seed(int(time.time()))
# mx.random.seed(int(time.time()))


class NeuralNet(object):
    def __init__(self):
        self.num_class = 10
        self.learning_rate = 0.1

        loss = "MSE"
        self.loss = self.loss_func(loss)
        self.error = self.error_func(loss)

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = np.reshape(x_train, (-1, 28 * 28))
        x_test = np.reshape(x_test, (-1, 28 * 28))

        self.X = np.array(np.append(x_train, x_test, axis=0))
        self.Y = np.eye(self.num_class)[np.append(y_train, y_test)]  # one hot vectors

        self.model = []
        self.build_model()

    def __del__(self):
        pass

    def build_model(self):
        print("Build the model...\n")
        self.model.append(Layer("fc", len(self.X[0]), 256, self.learning_rate, "sigmoid"))
        self.model.append(Layer("fc", 256, 10, self.learning_rate, "sigmoid"))

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

        with open(os.path.join('weights_512_MSE.txt'), 'wb') as fp:
            pickle.dump(weights, fp)

    def evaluate(self):
        """Evaluate the model."""
        global_loss = 0
        predicted_values = []

        for i in range(len(self.X)):

            # forward
            o = self.forward(i)

            loss, predicted_value = self.loss(o, self.Y[i])
            predicted_values.append(predicted_value)

            global_loss += loss

        return global_loss, self.accurate_func(np.array(predicted_values))

    def train_step(self):
        """Train one epoch on the network with backpropagation."""

        # alternative async

        for i in range(len(self.X)):

            # forward
            o = self.forward(i)

            error = self.error(o, self.Y[i])

            # backward
            self.backward(error, i)

    def forward(self, i):
        data = self.X[i]
        for layer in self.model:
            data = layer.forward(data)
        return data

    def backward(self, error, i):
        output_bp = error
        for j in range(len(self.model))[::-1]:
            layer = self.model[j]
            if j == len(self.model) - 1 and len(self.model) != 1:
                output_bp = layer.backward(self.model[j - 1].output, error)
            elif j == 0:
                output_bp = layer.backward(self.X[i], output_bp)
            else:
                output_bp = layer.backward(self.model[j - 1].output, output_bp)

    def base_line(self, epochs):
        print("Start training the model...\n")

        for i in range(epochs):
            start = time.time()
            self.train_step()
            loss_value, accurate = self.evaluate()
            # print("max of W: {0:.2f}".format(np.amax(self.W)))
            # print("min of W: {0:.2f}\n".format(np.amin(self.W)))
            # print("max of b: {0:.2f}".format(np.amax(self.b)))
            # print("min of b: {0:.2f}\n".format(np.amin(self.b)))

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


class Layer(object):
    def __init__(self, layer_type, input_size, output_size, learning_rate, activation):
        self.layer_type = layer_type
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.act = self.act_func(activation)
        self.d_act = self.d_act_func(activation)

        self.output = None

        self.W = None
        self.b = None
        self.z = None

        self.forward = None
        self.backward = None

        if self.layer_type == "fc":
            self.W = (np.random.rand(self.input_size, self.output_size) * 1) - 0.5
            self.b = (np.random.rand(self.output_size) * 1) - 0.5
            self.forward = self.dense_fw
            self.backward = self.dense_bw

    def dense_fw(self, x):
        """Fully connected layer forward process."""
        self.z = np.add(np.dot(x, self.W), self.b)
        self.output = self.act(self.z)
        return self.output

    def dense_bw(self, input_layer, input_error):
        """Fully connected layer backward process"""
        d_act_z = self.d_act(self.z)

        delta_b = np.multiply(input_error, d_act_z)
        delta_b = delta_b.reshape((1, -1))

        x = input_layer.reshape((1, -1))
        x = np.transpose(x)
        delta_W = np.dot(x, delta_b)

        output_bp = np.dot(delta_b, np.transpose(self.W))

        self.W = np.subtract(self.W, delta_W * self.learning_rate)
        self.b = np.subtract(self.b, delta_b * self.learning_rate)

        return output_bp

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
        return np.tanh(x)

    def d_tanh(self, x):
        return 1 - np.tanh(x) ** 2

    def log(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def d_log(self, x):
        return self.log(x) * (1 - self.log(x))

    def relu(self, x):
        return np.maximum(x, 0)

    def d_relu(self, x):
        return np.where(x > 0, 1, 0)


if __name__ == "__main__":
    nn = NeuralNet()
    nn.base_line(60)
    nn.save_weights()




