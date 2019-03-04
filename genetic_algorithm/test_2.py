from datetime import datetime
from matplotlib.animation import FuncAnimation
from random import randrange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pickle
import os


def reload_dict():
    with open(os.path.join("/home/biot/projects/research/logs", "log"), "rb") as log_file:
        logs = pickle.load(log_file)

    log_dict = dict()
    for log in logs:
        for key, inner_dict in log.items():
            for inner_key, item in inner_dict.items():
                if key + '/' + inner_key not in log_dict.keys():
                    log_dict[key + '/' + inner_key] = []
                log_dict[key + '/' + inner_key] += [item]

    keys = list(log_dict.keys())
    valid_keys = list(filter(lambda key: key.split('/')[1] != "iteration", keys))
    xdata = log_dict["iteration_end/iteration"]

    return log_dict, valid_keys, xdata


class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1

    def prev(self, event):
        self.ind -= 1


axprev = plt.axes([0.7, 0.01, 0.1, 0.075])
axnext = plt.axes([0.81, 0.01, 0.1, 0.075])


x_data, y_data = [], []

fig, ax = plt.subplots()
l, = plt.plot(x_data, y_data, lw=2)


def update(frame):
    log_dict, valid_keys, xdata = reload_dict()
    x_data = xdata
    y_data = log_dict[valid_keys[callback.ind % len(valid_keys)]]
    max_length = min(len(x_data), len(y_data))
    l.set_data(x_data[:max_length], y_data[:max_length])
    ax.set_title(valid_keys[callback.ind % len(valid_keys)])
    fig.gca().relim()
    fig.gca().autoscale_view()
    return l,


callback = Index()
animation = FuncAnimation(fig, update, interval=200)

bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)


plt.show()
