import numpy as np
from datatype.abstract_db import AbstractDB
from resources.datasets import load_raw
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import make_pipeline

class Word2VecDB(AbstractDB):

    def __init__(self):
        AbstractDB.__init__(self)
        db = load_raw()

        vectorizer = CountVectorizer()
        pipe = make_pipeline(vectorizer, TfidfTransformer())
        pipe.fit(db['text'])

        self.__input = pipe.transform(db['text'])
        self.__input_labels = np.array(list(sorted(vectorizer.vocabulary_.keys(),
                                                 key=lambda k: vectorizer.vocabulary_[k])))

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