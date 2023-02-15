from pymongo import MongoClient
import pandas as pd

filename = 'data/submit.csv'
#print(csv_file.head())
#print(csv_file.shape)

client = MongoClient("mongodb://rootuser:rootpass@localhost:27017/")

db = client["fakenews"]
collection = db["submit"]

df = pd.read_csv(filename)

for index, row in df.iterrows():
    article_id = int(row['id'])
    #print(type(article_id))
    label = str(row['label'])
    document = {"_id": article_id, "label": label}
    #print(document)
    collection.insert_one(document)     
