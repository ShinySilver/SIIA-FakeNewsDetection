from abc import ABC, abstractmethod


class AbstractDB(ABC):

    @abstractmethod
    def input_shape(self):
        pass

    @abstractmethod
    def input(self):
        pass

    @abstractmethod
    def input_labels(self):
        pass

    @abstractmethod
    def output(self):
        pass

    @abstractmethod
    def output_labels(self):
        pass