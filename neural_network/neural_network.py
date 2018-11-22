import numpy as np
import tensorflow as tf
import time
import pickle
import os
from layer import Input, Dense, Conv2d, Pool2d, Flatten, Concat, Add

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
        self.epochs = None

        self.input = input
        self.output = output

        self.model = None

    def __del__(self):
        pass

    def build_model(self, loss="MSE", learning_rate=0.1, batch_size=100):
        print("Build the model...\n")
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.learning_rate = learning_rate
        self.loss = self.loss_func(loss)
        self.error = self.error_func(loss)
        self.input.set_size_forward(self.batch_size, self.learning_rate)

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
            # print(b)

            # forward process
            start, end = b * self.batch_size, (b + 1) * self.batch_size
            self.forward(start, end)
            o = self.output.output

            loss, predicted_value = self.loss(o, self.Y[start:end])
            # print(loss)
            # print(predicted_value.shape)
            # exit()
            predicted_values.append(predicted_value)

            global_loss += loss

        predicted_values = np.array(predicted_values).reshape(-1,)

        return global_loss, self.accurate_func(np.array(predicted_values))

    def train_step(self):
        """Train one epoch on the network with backpropagation."""

        for b in range(self.num_batches):
            # print(b)

            # forward
            start_time = time.time()
            start, end = b * self.batch_size, (b + 1) * self.batch_size
            self.forward(start, end)
            o = self.output.output

            # print("Time of forward: {}s".format(time.time() - start_time))
            error = self.error(o, self.Y[start:end])

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

    def train(self, X, Y, epochs):
        self.X = X
        self.Y = Y
        self.Y_am = np.argmax(Y.reshape(-1, 10), axis=1)

        self.epochs = epochs
        self.num_batches = self.X.shape[0] // self.batch_size

        print("Start training the model...\n")

        for i in range(self.epochs):
            start = time.time()
            self.train_step()
            loss_value, accurate = self.evaluate()

            print("EPOCH", i + 1, "\tAccurate: {0:.2f}%\t".format(accurate * 100), "Loss: {0:.4f}\t".format(loss_value), "Time: {0:.2f}s\n".format(time.time() - start))
            if i == 30:
                self.learning_rate *= 0.5

    def accurate_func(self, pred):
        goal = 0

        for i in range(pred.shape[0]):
            if pred[i] == self.Y_am[i]:         # shape: (70000, 1, 10) --> shape: (70000, 10)
                goal += 1

        return goal / pred.shape[0]

    def loss_func(self, type):
        def mse(o, y):
            return np.square(o - y).sum() * 0.5 / self.batch_size, np.argmax(o, axis=2)

        def xe(o, y):
            prediction = self.softmax(o)
            return self.cross_entropy(prediction, y), np.argmax(prediction, axis=2)

        if type == "MSE":
            return mse

        elif type == "XE":
            return xe

    def error_func(self, type):
        def mse(o, y):
            return np.subtract(o, y)

        def xe(o, y):
            """
            d_cross_entropy
            """
            prediction = self.softmax(o)
            return np.subtract(prediction, y)

        if type == "MSE":
            return mse

        elif type == "XE":
            return xe

    def softmax(self, x):
        """
        Compute softmax values for each sets of scores in x.
        https://deepnotes.io/softmax-crossentropy
        Input and Return shape: (batch size, num of class)
        """
        e_x = np.exp(x - np.max(x, axis=2).reshape(self.batch_size, 1, 1))
        ex_sum = np.sum(e_x, axis=2)
        ex_sum = ex_sum.reshape((self.batch_size, 1, 1))

        return e_x / ex_sum

    @staticmethod
    def cross_entropy(p, y):
        """
        p is the prediction after softmax, shape: (batch size, num of class)
        y is labels (one hot vectors), shape: (batch size, num of class)
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        https://deepnotes.io/softmax-crossentropy
        Return size is a scalar.
        cost = -(1.0/m) * np.sum(np.dot(np.log(A), Y.T) + np.dot(np.log(1-A), (1-Y).T))
        """
        # y = y.argmax(axis=1)
        m = y.shape[0]
        #
        # log_likelihood = -np.log(p[range(m), y])
        # loss = np.sum(log_likelihood) / m

        cost = -(1.0 / m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return cost

    @staticmethod
    def d_cross_entropy(p, y):
        """
        p is the prediction after softmax, shape: (batch size, num of class)
        y is labels (one hot vectors), shape: (batch size, num of class)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        https://deepnotes.io/softmax-crossentropy
        Return shape: (batch size, num of class)
        """
        grad = np.subtract(p, y)
        return grad


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
    # x_train = np.reshape(x_train, (-1, 1, 28, 28))
    # x_test = np.reshape(x_test, (-1, 1, 28, 28))

    X = np.array(np.append(x_train, x_test, axis=0))
    Y = np.eye(num_class)[np.append(y_train, y_test)].reshape(-1, 1, 10)  # one hot vectors shape: (70000, 1, 10)

    ip = Input(input_size=(1, 784))
    # x = Conv2d(number_of_kernel=3, kernel_size=5, activation="relu")(ip)
    # x = Pool2d(kernel_size=5)(x)
    # y = Conv2d(number_of_kernel=3, kernel_size=5, activation="relu")(ip)
    # y = Pool2d(kernel_size=5)(y)
    # a = Add(weights_of_layers=[1, 3])([x, y])
    # # c = Concat(axis=1)([x, y])
    # f = Flatten()(a)
    # x1 = Dense(units=50, activation="sigmoid")(f)
    # y1 = Dense(units=20, activation="sigmoid")(x1)
    # y2 = Dense(units=20, activation="sigmoid", learning_rate=1)(x1)
    # c1 = Concat(axis=1)([y1, y2])
    #
    # x2 = Dense(units=50, activation="sigmoid", learning_rate=1)(ip)
    # z1 = Dense(units=20, activation="sigmoid", learning_rate=1)(x2)
    # z2 = Dense(units=20, activation="sigmoid", learning_rate=1)(x2)
    # c2 = Concat(axis=1)([z1, z2])

    # c = Concat(axis=1)([c1, c2])
    op = Dense(units=num_class, activation="sigmoid")(ip)

    nn = NeuralNet(ip, op)
    nn.build_model(loss="XE", learning_rate=0.1, batch_size=100)
    nn.train(X[:10000], Y[:10000], epochs=25)
    # nn.save_weights()

    # TODO: optimizers
    # TODO: batch size in layers
    # TODO: save weigths

