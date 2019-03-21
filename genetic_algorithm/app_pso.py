import yaml
import datetime
from fitness_functions.fitness_functions import RastriginFunction, FullyConnected
from elements.callbacks import LogToFile, RemoteControl, SaveResult, CheckPoint, DimReduction

from algorithms.pso_alg import ParticleSwarm

path = "./logs/pso-{}".format(datetime.datetime.now().strftime('%y-%d-%m-%H-%M-%-S'))

with open("config_tmp.yml", 'r') as config_file:
    config = yaml.load(config_file)

ps = ParticleSwarm(**config)

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

dr = DimReduction(log_dir=path, dimensions=2, frequency=10)
callback_list.append(dr)

ps.compile(config=config, fitness_function=fcf, callbacks=callback_list)

print("Run particle swarm algorithm\n")

ps.run()

# TODO: own config file, flags

