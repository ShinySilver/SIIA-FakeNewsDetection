from abc import ABC,abstractmethod



class AbstractClassifier(ABC):
    
    def __init__(self):
        pass

    @abstractmethod
    def train(self,*kargs,**kwargs):
        pass

    @abstractmethod
    def predict(self,*kargs,**kwargs):
        pass


