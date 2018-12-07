import numpy as np
import time
import pickle
import os

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

    def save_weights(self, filepath):
        weights = []
        self.input.save_weights(weights)

        with open(os.path.join(filepath), 'wb') as fp:
            pickle.dump(weights, fp)

    def load_weights(self, filepath):
        with open(os.path.join(filepath), 'rb') as fp:
            weights = pickle.load(fp)

        weights = np.array(weights)
        self.input.load_weights(weights)

    def predict(self, sample):
        """
        sample must have shape (1, 1, -1)
        """
        self.input.forward_process(sample)
        o = self.output.output
        prediction = self.softmax(o)

        return prediction

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
    print("Please use the train.py!")

    # TODO: optimizers


