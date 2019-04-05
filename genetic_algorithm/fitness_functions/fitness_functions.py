import numpy as np
import math
import time
from abc import ABC, abstractmethod
import os
import GPUtil

from fitness_functions.fully_connected_nn import train_model


class FitnessFunctionBase(ABC):
    """Base class of fitness function for metaheuristic algorithms."""

    def calculate(self, genes):
        phenotype_of_genes = self.genotype_to_phenotype(genes)
        return self.fitness_function(phenotype_of_genes)

    @abstractmethod
    def fitness_function(self, phenotype_of_genes):
        """Calculate fitness value from the genes."""
        pass

    @abstractmethod
    def genotype_to_phenotype(self, genes):
        """Transform genotype (genes) to phenotype for the fitness function."""
        pass


class RastriginFunction(FitnessFunctionBase):
    """https://en.wikipedia.org/wiki/Rastrigin_function"""

    def fitness_function(self, phenotype_of_genes):
        n = len(phenotype_of_genes)
        return 10 * n + sum([i * i - 10 * np.cos(2 * i * np.pi) for i in phenotype_of_genes])

    def genotype_to_phenotype(self, genes):
        phenotype_of_genes = (np.array(genes) * 10.24) - 5.12
        return phenotype_of_genes


class FullyConnected(FitnessFunctionBase):
    """Train a fully connected neural network on mnist from the fully_connected_nn.py."""

    def fitness_function(self, phenotype_of_genes):
        # num_of_gpus = 4
        #
        # time.sleep(np.random.random() * 4)
        # current_gpu = 0  # os.getpid() % num_of_gpus
        #
        # while True:
        #     available_gpus = GPUtil.getAvailable(order='first', limit=8, maxLoad=0.1, maxMemory=1, includeNan=False, excludeID=[], excludeUUID=[])
        #     if current_gpu not in available_gpus:
        #         current_gpu = (current_gpu + 1) % num_of_gpus
        #         # time.sleep(2)
        #     else:
        #         break
        #
        # phenotype_of_genes["gpu"] = [current_gpu]

        # max_num_of_params = 21620815, 6 hidden layers
        result, num_of_params = train_model(phenotype_of_genes)
        val_acc_ratio = 100 - (result * 100)  # [1, 100]
        num_of_params_ratio = np.log10(num_of_params)  # [3.89, 7.34]
        print("Number of parameters: {} / {}\nResult: {} / {}\n\n".format(num_of_params, num_of_params_ratio, result, val_acc_ratio))

        return val_acc_ratio + (num_of_params_ratio / 5)

    def genotype_to_phenotype(self, genes):
        # TODO: param min max

        genes = np.array(genes)
        input_dict = {"gpu": [0]}

        num_of_hidden_layers = int(math.floor(genes[0] * 4.999))
        input_dict["num_of_hidden_layers"] = num_of_hidden_layers

        size_of_layers = np.floor((genes[1:5] * 1490) + 5).astype("int")
        input_dict["size_of_layers"] = size_of_layers

        dropouts = genes[5:10] * 0.9
        input_dict["dropouts"] = dropouts

        learning_rate = np.power(10, -4 * genes[10])
        input_dict["learning_rate"] = learning_rate

        return input_dict


