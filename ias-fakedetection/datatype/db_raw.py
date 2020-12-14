import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from datatype.abstract_db import AbstractDB
from resources.datasets import load_raw


class RawDB(AbstractDB):

    def __init__(self):
        AbstractDB.__init__(self)

        db = load_raw()

        self.__input = np.array(db['text'])
        self.__input_labels = None

        self.__output = db['class_id']
        self.__output_labels = db['labels']

    def input_shape(self):
        return self.__input.shape

    def input(self):
        return self.__input

    def input_labels(self):
        return self.__input_labels

    def output(self):
        return self.__output

    def output_labels(self):
        return self.__output_labels