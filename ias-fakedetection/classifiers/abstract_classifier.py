from import ABC,abstractmethod



class AbstractClassifier(ABC):
    
    def __init__(self,data):
        self.data=data    



    @abstractmethod
    def train(test_prop=0.3):
        pass

    @abstractmethod
    def predict(new_data):
        pass




