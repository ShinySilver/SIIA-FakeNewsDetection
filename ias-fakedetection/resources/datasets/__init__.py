import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'news_fake_or_real.csv')

_DB_RAW = None
_DB_BOW = None

def load_raw():
    global _DB_RAW
    if _DB_RAW is None:
        db = dict(pd.read_csv(DATASET_PATH))
        db['label_id']=[int(label=='REAL') for label in db['label']]
        db['labels']=['FAKE','REAL']
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
        db['dictionnary'] = vectorizer.vocabulary_
        db['indexed_text'] = vectorizer.transform(db['text'])
        _DB_BOW = db
        return db
    else:
        return _DB_BOW