import yaml
from elements.fitness_functions import RastriginFunction
from elements.callbacks import LogToFile, RemoteControl

from algorithms.gen_alg import GeneticAlgorithm

# TODO: MethaBoard

with open("config.yml", 'r') as config_file:
    config = yaml.load(config_file)

ga = GeneticAlgorithm(**config)

ff = RastriginFunction()

callback_list = []
ltf = LogToFile(file_path="/home/biot/projects/research/logs")
callback_list.append(ltf)

rc = RemoteControl(config_file="config.yml")
callback_list.append(rc)

ga.compile(config=config,
           fitness_function=ff,
           callbacks=callback_list)

print("Run genetic algorithm\n")

for key, item in config.items():
    if item is not None:
        print("{0}: {1}".format(key, item))
print('\n')

ga.run()

