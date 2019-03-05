import numpy as np
from abc import ABC, abstractmethod


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

