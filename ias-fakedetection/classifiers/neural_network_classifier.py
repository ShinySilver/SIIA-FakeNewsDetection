from  classifiers.abstract_classifier import AbstractClassifier
from keras import models,layers


class NeuralNetworkClassifier(AbstractClassifier):
    
    def __init__(self,lengthData):
        self.__model=models.Sequential()
        self.__model.add(layers.Dense(16, activation='relu',
                                      input_shape=(lengthData,)))
        self.__model.add(layers.Dense(16, activation='relu'))
        self.__model.add(layers.Dense(1, activation='sigmoid'))
        self.__model.compile(optimizer='rmsprop',
                             loss='binary_crossentropy',
                             metrics=['accuracy'])
        
        
    def train(self,*kargs,**kwargs):
        return self.__model.fit(*kargs,**kwargs)
    def predict(self,*kargs,**kwargs):
        return self.__model.predict(*kargs,**kwargs)
