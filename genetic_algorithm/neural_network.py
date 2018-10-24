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
        self.loss_func = "MSE"

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = np.reshape(x_train, (-1, 28 * 28))
        x_test = np.reshape(x_test, (-1, 28 * 28))

        self.X = np.append(x_train, x_test, axis=0)
        self.Y = np.eye(self.num_class)[np.append(y_train, y_test)]  # one hot vectors

        self.W = (np.random.rand(784, 10) * 2) - 1
        self.b = (np.random.rand(10) * 0.5) - 1

    def __del__(self):
        pass

    def set_W(self, tensor):
        pass     # shape (784, 10)

    def set_b(self, vector):
        pass         # shape (10,)

    def evaluate(self, acc=False):

        if acc:
            accurate = self.accurate_func(predicted)

            #sess.close()
            return loss, accurate
        else:
            #sess.close()
            return loss, 0

    # one epoch
    def train_step(self):
        global_loss = 0
        predicted_values = []

        # def one_train_sample(indexes):
        #     loss = 0
        #     predicted_value = []
        #     delta_W = np.zeros((784, 10))
        #     delta_b = np.zeros((1, 10))
        #
        #     for i in indexes:
        #         # forward
        #         z = np.add(np.dot(self.X[i], self.W), self.b)
        #         o = self.log(z)
        #         loss_in = 0
        #
        #         if self.loss_func == "MSE":
        #             loss_in = np.square(o - self.Y[i]).sum() * 0.5
        #             predicted_value.append(np.argmax(o))
        #             error = np.subtract(o, self.Y[i])
        #         if self.loss_func == "XE":
        #             loss_in = self.cross_entropy(o, self.Y[i])
        #             predicted_value.append(np.argmax(self.softmax(o)))
        #             error = self.d_cross_entropy(o, self.Y[i])[0]
        #
        #         loss += loss_in
        #         # backward
        #         d_log_z = self.d_log(z)
        #         delta_b_in = np.multiply(error, d_log_z)
        #         delta_b = np.add(delta_b, delta_b_in.reshape((1, -1)))
        #         X = self.X[i].reshape((1, -1))
        #         X = np.transpose(X)
        #         delta_W = np.add(delta_W, np.dot(X, delta_b_in.reshape((1, -1))))
        #
        #     print(loss)
        #     return loss, predicted_value, delta_W, delta_b
        #
        # print('Use process pool for local search with pool size {}.'.format(cpu_count()))
        # epoch_index = list(range(len(self.X)))
        # p = Pool(cpu_count())
        # batch_size = 10240
        #
        # for i in range(len(self.X) // (batch_size * cpu_count())):
        #     start = i * (batch_size * cpu_count())
        #     end = (i + 1) * (batch_size * cpu_count())
        #
        #     if end > len(self.X):
        #         end = len(self.X)
        #
        #     batch = epoch_index[start:end]
        #     batches = []
        #
        #     for j in range(cpu_count()):
        #         b_start = j * batch_size
        #         b_end = (j + 1) * batch_size
        #         if b_end > end:
        #             b_end = end
        #         batches.append(batch[b_start:b_end])
        #
        #     print(batches[-1][-1])
        #     results = p.map(one_train_sample, batches)
        #
        #     for loss, predicted_value, delta_W, delta_b in results:
        #         global_loss += loss
        #         predicted_values = predicted_values + predicted_value
        #         self.W = self.W - delta_W * self.learning_rate
        #         self.b = self.b - delta_b * self.learning_rate
        #
        # p.terminate()
        # return global_loss, np.array(predicted_values)

        for i in range(len(self.X)):

            # forward
            z = np.add(np.dot(self.X[i], self.W), self.b)
            o = self.log(z)

            if self.loss_func == "MSE":
                loss = np.square(o - self.Y[i]).sum() * 0.5
                predicted_values.append(np.argmax(o))
                error = np.subtract(o, self.Y[i])
            if self.loss_func == "XE":
                loss = self.cross_entropy(o, self.Y[i])
                predicted_values.append(np.argmax(self.softmax(o)))
                error = self.d_cross_entropy(o, self.Y[i])[0]

            global_loss += loss

            # backward
            d_log_z = self.d_log(z)
            delta_b = np.multiply(error, d_log_z)
            delta_b = delta_b.reshape((1, -1))
            X = self.X[i].reshape((1, -1))
            X = np.transpose(X)
            delta_W = np.dot(X, delta_b)

            self.W = np.subtract(self.W, delta_W * self.learning_rate)
            self.b = np.subtract(self.b, delta_b * self.learning_rate)

        return global_loss, np.array(predicted_values)

    def base_line(self, epochs):

        for i in range(epochs):
            start = time.time()
            loss_value, predicted = self.train_step()
            print("EPOCH", i + 1, "\tAccurate: {0:.2f}%\t".format(self.accurate_func(predicted) * 100), "Loss: {0:.2f}\t".format(loss_value), "ETA: {0:.2f}s\n".format(time.time() - start))
            if i > 60:
                self.learning_rate *= 0.5

    def accurate_func(self, pred):
        goal = 0

        for i in range(pred.shape[0]):

            if pred[i] == np.argmax(self.Y[i]):
                goal += 1
        return goal / pred.shape[0]

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def cross_entropy(self, X, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        y = np.array(y).reshape((1, -1))
        X = np.array(X).reshape((1, -1))
        y = y.argmax(axis=1)
        m = y.shape[0]
        p = self.softmax(X)
        # We use multidimensional array indexing to extract
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def d_cross_entropy(self, X, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        y = np.array(y).reshape((1, -1))
        X = np.array(X).reshape((1, -1))
        y = y.argmax(axis=1)
        m = y.shape[0]
        grad = self.softmax(X)
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


nn = NeuralNet()

nn.base_line(100)
#
# W = np.ones(shape=(784, 10))
# b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# #nn.set_b(vector=b)
# #nn.set_w(W)
#
# e = nn.evaluate()
# print(e)
#
# nn.sess.close()


