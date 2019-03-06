from abc import ABC
import os
import yaml
import pickle
from multiprocessing import cpu_count

from elements.selections import selection_functions
from elements.memetics import memetic_functions
from elements.mutations import mutation_functions


class CallbackBase(ABC):
    """Base class of callbacks for metaheuristic algorithms."""

    def __init__(self):
        self.model = None

    def on_search_begin(self, logs):
        pass

    def on_search_end(self, logs):
        pass

    def on_iteration_begin(self, logs):
        pass

    def on_iteration_end(self, logs):
        pass

    def on_step_begin(self, logs):
        pass

    def on_step_end(self, logs):
        pass

    def set_model(self, model):
        self.model = model


class LogToFile(CallbackBase):
    """Write log to a given file."""

    def __init__(self, file_path="./logs"):
        super().__init__()

        self.file_path = file_path

        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)

    def on_iteration_end(self, logs):
        with open(os.path.join(self.file_path, "log"), "wb+") as log_file:
            pickle.dump(logs, log_file)

    def on_search_end(self, logs):
        with open(os.path.join(self.file_path, "log"), "wb+") as log_file:
            pickle.dump(logs, log_file)


class RemoteControl(CallbackBase):
    """Recompile the model from a given config file."""

    def __init__(self, config_file):
        super().__init__()

        self.config_file = config_file

    def on_iteration_end(self, logs):
        """Recompile the functions of the algorithm."""

        with open(self.config_file, 'r') as config_file:
            self.model.config = yaml.load(config_file)

        if self.model.config["active"] is True:
            self.model.selection_function = selection_functions(**self.model.config)
            self.model.mutation_function = mutation_functions(**self.model.config)
            self.model.memetic_function = memetic_functions(**self.model.config)
            self.model.init_steps()

            self.model.stop = self.model.config["stop"]
            self.model.pool = self.model.config["pool"]
            self.model.pool_size = cpu_count() if self.model.config["pool_size"] is None else self.model.config["pool_size"]
            self.model.elitism = False if self.model.config["elitism"] is None else self.model.config["elitism"]
            self.model.num_of_new_individual = self.model.population_size // 2 if self.model.config["num_of_new_individual"] is None else self.model.config["num_of_new_individual"]
            self.model.num_of_crossover = self.model.population_size // 4 if self.model.config["num_of_crossover"] is None else self.model.config["num_of_crossover"]




