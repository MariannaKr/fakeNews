#import libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
import time

news_data = pd.read_csv('data/train.csv')
print(news_data.head())
print(news_data.shape)

# counting the number of missing values in the dataset
news_data.isnull().sum()

# replacing the null values with empty string
news_data = news_data.fillna('')

# checking the number of missing values in the dataset
news_data.isnull().sum()

# merging the author name and news title
news_data['content'] = news_data['author']+' '+news_data['title']+' '+news_data['text']
print(news_data['content'])

#stemming
nltk.download('stopwords')
port_stem = PorterStemmer()

def stemming(content):
    review = re.sub('[^a-zA-Z]',' ',content)
    review = review.lower()
    review = review.split()
    review = [port_stem.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

news_data['content'] = news_data['content'].apply(stemming)
print(news_data['content'])

#separating the data and label
X = news_data['content'].values
Y = news_data['label'].values

print(X)
print(Y)

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

joblib.dump(vectorizer, 'saved_models/vectorizer.pkl')

"""## Classification """

#split train & test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=30)
X_test.shape

"""### Classifier 1"""

start = time.time()
logRegression = LogisticRegression()
logRegression.fit(X_train, Y_train)
end = time.time()

time_to_train_lg = float(end - start)

start = time.time()
lg_predictions = logRegression.predict(X_test)
end = time.time()

time_to_predict_lg = float(end - start)

lg_accuracy = accuracy_score(lg_predictions, Y_test)

joblib.dump(logRegression, 'saved_models/logRegression.pkl')


print('Accuracy score of Logistic Regression : ', "%.3f" % lg_accuracy)
print('Time to train Logistic Regression : ', "%.6f" % time_to_train_lg, " seconds")
print('Time for predictions with Logistic Regression : ', "%.6f" % time_to_predict_lg, " seconds")

"""### Classifier 2"""

start = time.time()
svm = SVC(C=1.0, gamma=0.1)
svm.fit(X_train, Y_train)
end = time.time()

time_to_train_svm = float(end - start)

start = time.time()
svm_predictions = svm.predict(X_test)
end = time.time()

time_to_predict_svm = float(end - start)

svm_accuracy = accuracy_score(svm_predictions, Y_test)

joblib.dump(svm, 'saved_models/svm.pkl')

print('Accuracy score of SVM : ', "%.3f" % svm_accuracy)
print('Time to train SVM : ', "%.6f" % time_to_train_svm, " seconds")
print('Time for predictions with SVM : ', "%.6f" % time_to_predict_svm, " seconds")