import yaml
from fitness_functions.fitness_functions import RastriginFunction, FullyConnected
from elements.callbacks import LogToFile, RemoteControl, SaveResult

from algorithms.pso_alg import ParticleSwarm

with open("config_tmp.yml", 'r') as config_file:
    config = yaml.load(config_file)

ps = ParticleSwarm(**config)

rf = RastriginFunction()
fcf = FullyConnected()

callback_list = []
ltf = LogToFile(file_path="./logs")  # "/home/biot/projects/research/logs")
callback_list.append(ltf)

rc = RemoteControl(config_file="config_tmp.yml")
callback_list.append(rc)

sr = SaveResult(result_file="./logs/result.txt", iteration_end=True)  # "/home/biot/projects/research/logs/result.txt")
callback_list.append(sr)

ps.compile(config=config,
           fitness_function=fcf,
           callbacks=callback_list)

print("Run particle swarm algorithm\n")

# TODO: print init and config from class

for key, item in config.items():
    if item is not None:
        print("{0}: {1}".format(key, item))
print('\n')

ps.run()

