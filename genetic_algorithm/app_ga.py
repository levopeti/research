import yaml
import time
import fitness_functions
from callbacks import LogToFile
import os
import numpy as np
import matplotlib.pyplot as plt

from gen_alg import GeneticAlgorithm

# TODO: LogsCallback, Fitness function class

with open("config.yml", 'r') as config_file:
    config = yaml.load(config_file)

ga = GeneticAlgorithm(**config)

ff = fitness_functions.FitnessFunction(1)

ltf = LogToFile(file_path="/home/biot/projects/research/logs")

ga.compile(config=config,
           fitness_function=ff,
           remote_config="config.yml",
           callbacks=None)

print("Run genetic algorithm\n")

for key, item in config.items():
    if item is not None:
        print("{0}: {1}".format(key, item))
print('\n')

ga.run()

