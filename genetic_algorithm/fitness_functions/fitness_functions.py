import numpy as np
import math
from abc import ABC, abstractmethod

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
        # max_num_of_params = 21620815

        result, num_of_params = train_model(phenotype_of_genes)
        val_acc_ratio = 100 - (result * 100)  # [1, 100]
        num_of_params_ratio = np.log10(num_of_params)  # [3.89, 7.34]

        return val_acc_ratio + (num_of_params_ratio / 2)

    def genotype_to_phenotype(self, genes):
        genes = np.array(genes)
        input_dict = {"gpu": [0]}

        num_of_hidden_layers = int(math.floor(genes[0] * 6.999))
        input_dict["num_of_hidden_layers"] = num_of_hidden_layers

        size_of_layers = np.floor((genes[1:7] * 1990) + 11).astype("int")
        input_dict["size_of_layers"] = size_of_layers

        dropouts = genes[7:14]
        input_dict["dropouts"] = dropouts

        learning_rate = np.power(10, -4 * genes[14])
        input_dict["learning_rate"] = learning_rate

        return input_dict


