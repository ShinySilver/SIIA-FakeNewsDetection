from  classifiers.abstract_classifier import AbstractClassifier
from keras import models,layers


class DeepNeuralClassifier(AbstractClassifier):
    
    def __init__(self,lengthData,networkShape=(16,16,16,16)):
        
        assert len(networkShape)>2
        self.__model=models.Sequential()
        
        self.__model.add(layers.Dense(networkShape[0], activation='relu',
                                      input_shape=(lengthData,)))
        
        for size in networkShape[1:]:
            self.__model.add(layers.Dense(size, activation='relu'))

        self.__model.add(layers.Dense(1, activation='sigmoid'))
        
        self.__model.compile(optimizer='rmsprop',
                             loss='binary_crossentropy',
                             metrics=['accuracy'])
        
        
    def train(self,*kargs,**kwargs):
        return self.__model.fit(*kargs,**kwargs)
    def predict(self,*kargs,**kwargs):
        return self.__model.predict(*kargs,**kwargs)
