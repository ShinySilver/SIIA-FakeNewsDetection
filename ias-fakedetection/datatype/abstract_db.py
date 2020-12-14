from abc import ABC, abstractmethod


class AbstractDB(ABC):

    def input_shape(self):
        return self.input().shape

    @abstractmethod
    def input(self):
        pass

    def X(self):
        return self.input()

    @abstractmethod
    def input_labels(self):
        pass

    def X_labels(self):
        return self.input_labels()

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