import yaml
from fitness_functions.fitness_functions import RastriginFunction, FullyConnected
from elements.callbacks import LogToFile, RemoteControl, SaveResult

from algorithms.gen_alg import GeneticAlgorithm

with open("config_tmp.yml", 'r') as config_file:
    config = yaml.load(config_file)

ga = GeneticAlgorithm(**config)

rf = RastriginFunction()
fcf = FullyConnected()

callback_list = []
ltf = LogToFile(file_path="/home/biot/projects/research/logs")
callback_list.append(ltf)

rc = RemoteControl(config_file="config_tmp.yml")
callback_list.append(rc)

sr = SaveResult(result_file="/home/biot/projects/research/logs/result.txt")
callback_list.append(sr)

ga.compile(config=config,
           fitness_function=fcf,
           callbacks=callback_list)

print("Run genetic algorithm\n")

for key, item in config.items():
    if item is not None:
        print("{0}: {1}".format(key, item))
print('\n')

ga.run()

