import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from datatype.abstract_db import AbstractDB
from resources.datasets import load_raw


class BagOfWordDB(AbstractDB):

    def __init__(self):
        AbstractDB.__init__(self)

        db = load_raw()
        vectorizer = CountVectorizer()
        vectorizer.fit(db['text'])

        self.__input = vectorizer.transform(db['text']).toarray()
        self.__input_labels = np.array(list(sorted(vectorizer.vocabulary_.keys(),
                                                 key=lambda k: vectorizer.vocabulary_[k])))

        self.__output = self


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