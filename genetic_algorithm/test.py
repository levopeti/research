import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pickle
import os

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

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.set_title(valid_keys[0])

ydata = np.array(log_dict[valid_keys[0]])
l, = plt.plot(xdata, ydata, lw=2)


class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        i = self.ind % len(valid_keys)
        ydata = np.array(log_dict[valid_keys[i]])
        ax.set_ylim(min(ydata) * 0.95, max(ydata) * 1.05)
        l.set_ydata(ydata)
        ax.set_title(valid_keys[i])
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(valid_keys)
        ydata = np.array(log_dict[valid_keys[i]])
        ax.set_ylim(min(ydata) * 0.95, max(ydata) * 1.05)
        l.set_ydata(ydata)
        ax.set_title(valid_keys[i])
        plt.draw()


callback = Index()
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()
