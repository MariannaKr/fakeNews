import pandas as pd
from apyori import apriori
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from langdetect import detect
import csv

nltk.download('punkt')

# Load dataset
df = pd.read_csv('data/onlyfake_train2K.csv')

# Text preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

df = df[df['text'].apply(lambda x: type(x)==str)]
df = df[df['text'].apply(lambda x: detect(x)=='en')]

df = df[df['text'].apply(lambda x: type(x)==str)]
df['tokenized_text'] = df['text'].apply(lambda x: [ps.stem(word) for word in word_tokenize(x) if word.isalpha()])
df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Convert the tokenized text to a list of lists
text_dataset = df['tokenized_text'].to_list()

articles=[]

for article in text_dataset:
  article= list( dict.fromkeys(article) )
  for word in article:
    if len(word)<=2:
      article.remove(word)
  articles.append(article)
  print(article)
  print("count of words: ",len(article))


min_support = 0.3

association_rules = apriori(articles, min_support=min_support)

output_file = 'ascosiationRules(0.3).csv'
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Rule', 'Support', 'Confidence', 'Lift', 'Leverage'])

    for item in association_rules:
        pair = item[0]
        items = [x for x in pair]
        if len(items) >= 2:
            rule = items[0] + ' -> ' + items[1]
            support = str(item[1])
            confidence = str(item[2][0][2])
            lift = str(item[2][0][3])
            leverage = str(item[2][0][3] - item[1])
            writer.writerow([rule, support, confidence, lift, leverage])