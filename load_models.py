import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def load():
    logRegression = joblib.load('saved_models/logRegression.pkl')
    svm = joblib.load('saved_models/svm.pkl')
    vectorizer = joblib.load('saved_models/vectorizer.pkl')
    return logRegression, svm, vectorizer

port_stem = PorterStemmer()

def stemming(content):
    review = re.sub('[^a-zA-Z]',' ',content)
    review = review.lower()
    review = review.split()
    review = [port_stem.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

def predict_category(text, vectorizer, logRegression, svm,title):
    text = stemming(text)
    X = vectorizer.transform([text])
    category_logRegression = logRegression.predict(X)[0]
    print(category_logRegression)
    category_svm = svm.predict(X)[0]
    print(category_svm)
    return title,category_logRegression,category_svm

