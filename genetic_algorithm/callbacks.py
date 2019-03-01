from abc import ABC
import os


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

    def __init__(self, file_path):
        super().__init__()

        self.file_path = file_path

        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)

    def on_iteration_end(self, logs):
        with open(os.path.join(self.file_path, "log.txt"), "a+") as log_file:
            log_file.write(str(logs))




