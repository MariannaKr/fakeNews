import pandas as pd
from apyori import apriori
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk


nltk.download('punkt')

# Load dataset
df = pd.read_csv('data/train.csv')

# Text preprocessing
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Tokenize and stem the articles
df = df[df['label'] == 1]
df = df[df['text'].apply(lambda x: type(x)==str)]
df['tokenized_text'] = df['text'].apply(lambda x: [ps.stem(word) for word in word_tokenize(x) if word.isalpha()])
df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Convert the tokenized text to a list of lists
text_dataset = df['tokenized_text'].to_list()

# Set the minimum support threshold
min_support = 0.01

# Generate association rules
association_rules = apriori(text_dataset, min_support=min_support)

# Create a list of tuples where each tuple contains the rule, support, confidence, and lift
results = []
for item in association_rules:
    pair = item[0]
    items = [x for x in pair]
    if len(items) >= 2:
        rule = items[0] + " -> " + items[1]
        support = item[1]
        confidence = item[2][0][2]
        lift = item[2][0][3]
        results.append((rule, support, confidence, lift))

# Create a DataFrame from the list of tuples
df = pd.DataFrame(results, columns=['Rule', 'Support', 'Confidence', 'Lift'])

# Write the DataFrame to a CSV file
df.to_csv('association_rules_small.csv', index=False)