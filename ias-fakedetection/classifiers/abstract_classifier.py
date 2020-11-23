from import ABC,abstractmethod



class AbstractClassifier(ABC):
    
    def __init__(self):

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def predict():
        pass


