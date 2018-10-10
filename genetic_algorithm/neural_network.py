import tensorflow as tf
import numpy as np
import time
import os


class NeuralNet(object):
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = np.reshape(x_train, (-1, 28 * 28))
        x_test = np.reshape(x_test, (-1, 28 * 28))

        x = np.append(x_train, x_test, axis=0)
        y_ = np.append(y_train, y_test)
        self.y_ = np.int64(y_)

        self.W = None
        self.b = None

        self.x = tf.constant(x)
        self.y = tf.constant(self.y_)

        self.gpu_counter = 0

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def __del__(self):
        self.sess.close()

    def set_w(self, tensor):
        self.W = tensor     # shape (784, 10)

    def set_b(self, vector):
        self.b = vector         # shape (10,)

    def evaluate(self, acc=False):
        #start = time.time()
        #print(self.gpu_counter)
        self.gpu_counter += 1
        #gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(int(self.gpu_counter)))

        #os.environ['CUDA_VISIBLE_DEVICES'] = str(int(self.gpu_counter))

        if self.gpu_counter == 3:
            self.gpu_counter = 0

        #with tf.device('/gpu:{}'.format(self.gpu_counter)):

        # sess = None
        b = False
        # loss = None
        # pred = None
        while b:
            try:
                b = False
                print(self.gpu_counter)
                gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(int(self.gpu_counter)))
                fc = tf.contrib.layers.fully_connected(self.x, 10, tf.nn.relu, weights_initializer=tf.constant_initializer(self.W), biases_initializer=tf.constant_initializer(self.b))
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=fc))
                pred = tf.argmax(fc, 1)
                sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

            except ValueError:
                #print(666)
                self.gpu_counter += 1
                if self.gpu_counter == 3:
                    self.gpu_counter = 0
                b = True

        fc = tf.contrib.layers.fully_connected(self.x, 10, tf.nn.relu, weights_initializer=tf.constant_initializer(self.W), biases_initializer=tf.constant_initializer(self.b))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=fc))
        pred = tf.argmax(fc, 1)

        self.sess.run(tf.global_variables_initializer())
        loss, predicted = self.sess.run([loss, pred])

        #end = time.time()
        #print('eval: ', end - start, 's')
        if acc:
            accurate = self.accurate_func(predicted)

            #sess.close()
            return loss, accurate
        else:
            #sess.close()
            return loss, 0

    def base_line(self, epochs):
        fc = tf.contrib.layers.fully_connected(self.x, 10, tf.nn.relu)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=fc))
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        pred = tf.argmax(fc, 1)

        tf.set_random_seed(1234)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(epochs):
                print('EPOCH', i + 1)
                _, predicted, loss_value = sess.run([train_op, pred, loss])
                #print("Loss: ", loss_value)
                print("Accurate: {0:.2f}%".format(self.accurate_func(predicted) * 100))
                print()

    def accurate_func(self, pred):
        goal = 0

        for i in range(pred.shape[0]):

            if pred[i] == self.y_[i]:
                goal += 1
        return goal / pred.shape[0]


# nn = NeuralNet()
#
# nn.base_line()
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


