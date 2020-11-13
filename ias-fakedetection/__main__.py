import resources.datasets as data

db_raw = data.load_raw()
print(db_raw)

db = data.load_BoW()
print(db)