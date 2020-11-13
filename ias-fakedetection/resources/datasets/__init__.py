import os
import pandas as pd

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'news_fake_or_real.csv')

DB = pd.read_csv(DATASET_PATH)
DB['label_id']=[int(label=='REAL') for label in DB['label']]