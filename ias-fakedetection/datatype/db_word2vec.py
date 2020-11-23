import numpy as np
from datatype.abstract_db import AbstractDB
from resources.datasets import load_raw


class Word2VecDB(AbstractDB):

    def __init__(self):
        AbstractDB.__init__(self)
        db = load_raw()

        self.__input = None
        self.__input_labels = None

        self.__output = db['class_id']
        self.__output_labels = db['labels']

    def input_shape(self):
        return (len(self.__input),)

    def input(self):
        return self.__input()

    def input_labels(self):
        return self.__input_labels

    def output(self):
        return self.__output

    def output_labels(self):
        return self.__output_labels