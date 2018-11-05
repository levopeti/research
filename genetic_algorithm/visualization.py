import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

with open(os.path.join('weights_XE.txt'), 'rb') as fp:
    weights = pickle.load(fp)

W = weights[0]
b = weights[1]

digits_hotmap = []

for i in range(10):
    digit = W[:, i]
    digit = np.reshape(digit, (28, 28))
    max_d = np.amax(digit)
    min_d = np.amin(digit)
    digit = (digit - min_d) / (max_d - min_d)
    digits_hotmap.append(digit)

for i, digit in enumerate(digits_hotmap):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digit)
    plt.title(i)

plt.show()


