import numpy as np
import time
import tensorflow as tf
from scipy import signal
from sklearn.metrics import log_loss
import os

np.random.seed(int(time.time()))


class NeuralNet(object):
    def __init__(self):
        self.num_class = 10
        self.learning_rate = 0.1

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = np.reshape(x_train, (-1, 28 * 28))
        x_test = np.reshape(x_test, (-1, 28 * 28))

        self.X = np.append(x_train, x_test, axis=0)
        self.Y = np.eye(self.num_class)[np.append(y_train, y_test)]  # one hot vectors

        self.W = (np.random.rand(784, 10) * 2) - 1
        self.b = (np.random.rand(10) * 0.56) - 1

    def __del__(self):
        pass

    def set_w(self, tensor):
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
        for i in range(len(self.X)):
            layer_fc = self.X[i].dot(self.W) + self.b
            layer_fc_act = self.log(layer_fc)
            softmax_v = self.softmax(layer_fc_act)

            loss = log_loss(self.Y[i], softmax_v)
            global_loss += loss
            predicted_values.append(np.argmax(softmax_v))

            grad_fc_part_1 = layer_fc_act - self.Y[i]
            grad_fc_part_2 = self.d_log(layer_fc)
            grad_fc_part_3 = self.X[i]
            grad_fc = grad_fc_part_3.T.dot(grad_fc_part_1 * grad_fc_part_2)

            self.W = self.W - grad_fc * self.learning_rate
            self.b = self.b - grad_fc_part_1 * self.learning_rate

            return global_loss, np.array(predicted_values)

    def base_line(self, epochs):

        for i in range(epochs):
                print('EPOCH', i + 1)
                loss_value, predicted = self.train_step()
                print("Accurate: {0:.2f}%".format(self.accurate_func(predicted) * 100))
                print()

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

    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        return 1 - np.tanh(x) ** 2

    def log(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def d_log(self, x):
        return self.log(x) * (1 - self.log(x))

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


