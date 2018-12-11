import numpy as np


class SGD(object):
    @staticmethod
    def run(W, velocity_W, delta_W, b, velocity_b, delta_b, learning_rate):
        W = np.subtract(W, delta_W * learning_rate)
        b = np.subtract(b, delta_b * learning_rate)

        return W, b, None, None


class Momentum(object):
    def __init__(self, gamma=0.9):
        self.gamma = gamma

    def run(self, W, velocity_W, delta_W, b, velocity_b, delta_b, learning_rate):
        new_velocity_W = np.add((self.gamma * velocity_W), (delta_W * learning_rate))
        new_velocity_b = np.add((self.gamma * velocity_b), (delta_b * learning_rate))

        W = np.subtract(W, new_velocity_W)
        b = np.subtract(b, new_velocity_b)

        return W, b, new_velocity_W, new_velocity_b





