"""
Optimization methods.
http://cs231n.github.io/neural-networks-3/
https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM
"""

import numpy as np


class SGD(object):
    """SGD with momentum and nesterov"""
    def __init__(self, gamma=0.9, nesterov=False):
        self.name = "SGD"
        self.gamma = gamma
        self.nesterov = nesterov

    def run(self, W, velocity_W, delta_W, b, velocity_b, delta_b, learning_rate):
        new_velocity_W = np.add((self.gamma * velocity_W), (delta_W * learning_rate))
        new_velocity_b = np.add((self.gamma * velocity_b), (delta_b * learning_rate))

        W = np.subtract(W, new_velocity_W)
        b = np.subtract(b, new_velocity_b)

        return W, b, new_velocity_W, new_velocity_b


class Adagrad(object):
    """
    Adaptive learning rate method.
    https://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
    It needs a smaller learning rate (0.1).
    """
    def __init__(self, eps=1e-6):
        self.name = "Adagrad"
        self.eps = eps

    def run(self, W, cache_W, delta_W, b, cache_b, delta_b, learning_rate):
        """Here velocity is a chace for pow(W, 2)."""

        cache_W += np.power(delta_W, 2)
        cache_b += np.power(delta_b, 2)

        adagrag_lr_W = learning_rate / (np.sqrt(cache_W) + self.eps)
        adagrag_lr_b = learning_rate / (np.sqrt(cache_b) + self.eps)

        W = np.subtract(W, adagrag_lr_W * delta_W)
        b = np.subtract(b, adagrag_lr_b * delta_b)

        return W, b, cache_W, cache_b


class Adadelta(object):
    pass


class RMSprop(object):
    """
    It needs a smaller learning rate (0.1).
    """
    def __init__(self, eps=1e-6, autocorr=0.95, first_order_momentum=False, momentum=False):
        self.name = "RMSprop"
        self.eps = eps
        self.autocorr = autocorr
        self.first_order_momentum = False
        self.momentum = False

    def run(self, W, cache_W, delta_W, b, cache_b, delta_b, learning_rate):
        """Here velocity is a chace for pow(W, 2)."""

        if not cache_W.any():
            """Only the first step."""
            cache_W = np.power(delta_W, 2)
            cache_b = np.power(delta_b, 2)
        else:
            cache_W = self.autocorr * cache_W + (1 - self.autocorr) * np.power(delta_W, 2)
            cache_b = self.autocorr * cache_b + (1 - self.autocorr) * np.power(delta_b, 2)

        adagrag_lr_W = learning_rate / (np.sqrt(cache_W) + self.eps)
        adagrag_lr_b = learning_rate / (np.sqrt(cache_b) + self.eps)

        W = np.subtract(W, adagrag_lr_W * delta_W)
        b = np.subtract(b, adagrag_lr_b * delta_b)

        return W, b, cache_W, cache_b


