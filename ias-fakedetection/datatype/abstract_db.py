from abc import ABC, abstractmethod


class AbstractDB(ABC):

    @abstractmethod
    def input_shape(self):
        pass

    @abstractmethod
    def input(self):
        pass

    def X(self):
        return self.input()

    @abstractmethod
    def input_labels(self):
        pass

    def X_labels(self):
        return input()

    @abstractmethod
    def output(self):
        pass

    def y(self):
        return self.output()

    @abstractmethod
    def output_labels(self):
        pass

    def y_labels(self):
        return self.output_labels()