import numpy as np
import tensorflow as tf
import time
from pathos.multiprocessing import Pool
from multiprocessing import cpu_count
from scipy import signal
from sklearn.metrics import log_loss
import os

np.random.seed(int(time.time()))


class NeuralNet(object):
    def __init__(self):
        self.num_class = 10
        self.learning_rate = 0.1
        self.loss = self.loss_func("MSE")

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = np.reshape(x_train, (-1, 28 * 28))
        x_test = np.reshape(x_test, (-1, 28 * 28))

        self.X = np.append(x_train, x_test, axis=0)
        self.Y = np.eye(self.num_class)[np.append(y_train, y_test)]  # one hot vectors

        self.W = (np.random.rand(784, 10) * 1) - 0.5
        self.b = (np.random.rand(10) * 1) - 0.5
        # print("max of W: {0:.2f}".format(np.amax(self.W)))
        # print("min of W: {0:.2f}\n".format(np.amin(self.W)))
        # print("max of b: {0:.2f}".format(np.amax(self.b)))
        # print("min of b: {0:.2f}\n".format(np.amin(self.b)))
        #
        # loss, acc = self.evaluate()
        # print("Accurate: {0:.2f}%\t".format(acc), "Loss: {0:.2f}\t".format(loss))
        # exit()

        self.z = 0

    def __del__(self):
        pass

    def set_weights(self, individual):
        self.W = np.reshape(np.array(individual[:7840]), (784, 10))  # shape (784, 10)
        self.b = np.array(individual[-10:])  # shape (10,)

    def get_weights_as_genes(self):
        return np.concatenate((np.reshape(self.W, (7840,)), self.b), axis=None)

    def evaluate(self):
        """Evaluate the model."""
        global_loss = 0
        predicted_values = []

        for i in range(len(self.X)):

            # forward
            o = self.dense_fw(self.X[i])

            loss, predicted_value, _ = self.loss(o, self.Y[i])
            predicted_values.append(predicted_value)

            global_loss += loss

        return global_loss, self.accurate_func(np.array(predicted_values))

    def train_step(self):
        """Train one epoch on the network with backpropagation."""

        # alternative async

        for i in range(len(self.X)):

            # forward
            o = self.dense_fw(self.X[i])

            _, _, error = self.loss(o, self.Y[i])

            # backward
            self.dense_bw(self.X[i], error)

    def base_line(self, epochs):

        for i in range(epochs):
            start = time.time()
            self.train_step()
            loss_value, accurate = self.evaluate()
            # print("max of W: {0:.2f}".format(np.amax(self.W)))
            # print("min of W: {0:.2f}\n".format(np.amin(self.W)))
            # print("max of b: {0:.2f}".format(np.amax(self.b)))
            # print("min of b: {0:.2f}\n".format(np.amin(self.b)))

            print("EPOCH", i + 1, "\tAccurate: {0:.2f}%\t".format(accurate), "Loss: {0:.2f}\t".format(loss_value), "ETA: {0:.2f}s\n".format(time.time() - start))
            if i > 60:
                self.learning_rate *= 0.5

    def accurate_func(self, pred):
        goal = 0

        for i in range(pred.shape[0]):

            if pred[i] == np.argmax(self.Y[i]):
                goal += 1
        return goal / pred.shape[0]

    def dense_fw(self, x):
        """Fully connected layer forward process."""
        self.z = np.add(np.dot(x, self.W), self.b)
        return self.log(self.z)

    def dense_bw(self, x, error):
        """Fully connected layer backward process"""
        d_log_z = self.d_log(self.z)
        delta_b = np.multiply(error, d_log_z)
        delta_b = delta_b.reshape((1, -1))
        x = x.reshape((1, -1))
        x = np.transpose(x)
        delta_W = np.dot(x, delta_b)

        self.W = np.subtract(self.W, delta_W * self.learning_rate)
        self.b = np.subtract(self.b, delta_b * self.learning_rate)

    def loss_func(self, type):
        def mse(o, y):
            return np.square(o - y).sum() * 0.5, np.argmax(o), np.subtract(o, y)

        def xe(o, y):
            return self.cross_entropy(o, y), np.argmax(self.softmax(o)), self.d_cross_entropy(o, y)[0]

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
    nn.base_line(30)




