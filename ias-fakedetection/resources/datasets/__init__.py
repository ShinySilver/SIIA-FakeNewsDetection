import os
import pandas as pd
import numpy as np
import re, string
from sklearn.feature_extraction.text import CountVectorizer

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'news_fake_or_real.csv')

_DB_RAW = None
_DB_BOW = None

def load_raw():
    global _DB_RAW
    if _DB_RAW is None:
        db = pd.read_csv(DATASET_PATH).to_dict(orient='list')

        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
        db['text'] =  [' '.join([re_punc.sub('', w) for w in text]) for text in db['text']]
        db['class_id']=np.array([int(label=='REAL') for label in db['label']])
        db['labels']=np.array(['FAKE','REAL'])
        _DB_RAW = db
        return db
    else:
        return _DB_RAW

def load_BoW():
    global _DB_BOW
    if _DB_BOW is None:
        db = load_raw()
        vectorizer = CountVectorizer()
        vectorizer.fit(db['text'])
        db['dictionnary'] = np.array(list(sorted(vectorizer.vocabulary_.keys(),
                                                 key=lambda k:vectorizer.vocabulary_[k])))
        db['text_bow'] = vectorizer.transform(db['text']).toarray()
        _DB_BOW = db
        return db
    else:
        return _DB_BOW