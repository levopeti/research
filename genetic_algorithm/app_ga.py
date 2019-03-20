import yaml
import datetime
from fitness_functions.fitness_functions import RastriginFunction, FullyConnected
from elements.callbacks import LogToFile, RemoteControl, SaveResult, CheckPoint

from algorithms.gen_alg import GeneticAlgorithm

path = "./logs/ga-{}".format(datetime.datetime.now().strftime('%y-%d-%m-%H-%M-%-S'))

with open("config_tmp.yml", 'r') as config_file:
    config = yaml.load(config_file)

ga = GeneticAlgorithm(**config)

rf = RastriginFunction()
fcf = FullyConnected()

callback_list = []
ltf = LogToFile(log_dir=path)
callback_list.append(ltf)

rc = RemoteControl(config_file="config_tmp.yml")
callback_list.append(rc)

sr = SaveResult(log_dir=path, iteration_end=True)
callback_list.append(sr)

cp = CheckPoint(log_dir=path, only_last=True)
callback_list.append(cp)

ga.compile(config=config, fitness_function=fcf, callbacks=callback_list)

print("Run genetic algorithm\n")

ga.run()

# TODO: own config file, flags

