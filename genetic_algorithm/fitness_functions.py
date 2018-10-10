import numpy as np
import keras
from neural_network import NeuralNet

from PIL import Image, ImageOps


class FitnessFunction(object):
    def __init__(self, id):
        self.id = id
        self.fitness_function = None

        self.image = None
        self.size = 120

        self.model = None

        if self.id == 1:
            self.fitness_function = self.fitness_func1

        if self.id == 2:
            image = Image.open('/home/biot/Pictures/kislevi.jpg')  # (960, 720, 3) -> 2073600       /home/biot/Pictures/kislevi.jpg     /root/inspiron/research/kislevi.jpg
            image = ImageOps.fit(image, (self.size, self.size), method=Image.ANTIALIAS)  # (50, 50, 3) -> 7500
            image = np.array(image)
            image = image.flatten()
            self.image = image / 255

            self.fitness_function = self.fitness_func2

        if self.id == 3:
            self.fitness_function = self.fitness_func3

        if self.id == 4:
            image = Image.open('/home/biot/Pictures/kislevi.jpg')  # (960, 720, 3) -> 2073600
            image = ImageOps.fit(image, (self.size, self.size), method=Image.ANTIALIAS)  # (50, 50, 3) -> 7500
            image = np.array(image)
            image = image.flatten()
            self.image = image / 255

            input = keras.Input(shape=(100, ))
            x = keras.layers.Dense(512)(input)
            x = keras.layers.Dense(self.size * self.size * 3)(x)

            model = keras.Model(input, x)
            self.model = model
            self.model.compile(loss='mean_squared_error', optimizer='sgd')

            self.fitness_function = self.fitness_func4

        if self.id == 5:
            self.model = NeuralNet()
            self.fitness_function = self.fitness_func5

    def calculate(self, individual, acc=False):
        return self.fitness_function(individual, acc=acc)

    @staticmethod
    def fitness_func1(individual, acc=False):
        return sum(individual.genes)

    def fitness_func2(self, individual, acc=False):
        dist = np.linalg.norm(individual.genes - self.image)
        return dist

    @staticmethod
    def fitness_func3(individual, acc=False):
        sum = 0
        for i, gene in enumerate(individual.genes):
            sum += abs(abs(np.sin(i / 20)) - gene)
        return sum

    def fitness_func4(self, individual, acc=False):
        #print(np.array([individual.genes]).shape)
        pred = self.model.predict(np.array([individual.genes]), batch_size=1)
        dist = np.linalg.norm(pred - self.image)
        return dist

    def fitness_func5(self, individual, acc=False):
        W = np.reshape(np.array(individual.genes[:7840]), (784, 10))
        b = np.array(individual.genes[-10:])
        self.model.set_w(W)
        self.model.set_b(b)
        loss, accurate = self.model.evaluate(acc)

        if acc:
            return loss, accurate
        else:
            return loss


