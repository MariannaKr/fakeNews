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
df['tokenized_text'] = df['text'].apply(lambda x: [ps.stem(word) for word in word_tokenize(x) if word.isalpha()])
df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Convert the tokenized text to a list of lists
text_dataset = df['tokenized_text'].to_list()

# Set the minimum support threshold
min_support = 0.01

# Generate association rules
association_rules = apriori(text_dataset, min_support=min_support)

# Print the extracted association rules
for item in association_rules:
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")