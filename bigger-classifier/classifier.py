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
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from scipy import sparse
import joblib
import time

train_data = pd.read_csv('./data/train.csv')
#data2 = pd.read_csv('./data/result.csv')
#train_data = pd.concat([data1, data2], axis=0)

#print(train_data.head())
#print(train_data.shape)

#print(test_data.head())
#print(test_data.shape)
# counting the number of missing values in the dataset
train_data.isnull().sum()
#test_data.isnull().sum()

# replacing the null values with empty string
train_data = train_data.fillna('')
#test_data = test_data.fillna('')s

# checking the number of missing values in the dataset
train_data.isnull().sum()
#test_data.isnull().sum()

# merging the author name and news title
train_data['content'] = train_data['author']+' '+train_data['title']+' '+train_data['text']
#print(train_data['content'])
#test_data['content'] = test_data['author']+' '+test_data['title']#+' '+test_data['text']
#print(test_data['content'])
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

train_data['content'] = train_data['content'].apply(stemming)
#test_data['content'] = test_data['content'].apply(stemming)
#print(train_data['content'])
print("finished stemming")
#separating the data and label
X = train_data['content'].values
Y = train_data['label'].values

#X_test = test_data['content'].values
#Y_test = test_data['label'].values


# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
print("finished vectorizing")
#vectorizer.fit(X_test)
#X_test = vectorizer.transform(X_test)
#joblib.dump(vectorizer, 'saved_models/vectorizer-no-text.pkl')

"""## Classification """

#split train & test data
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, stratify=Y, random_state=30)
#X_test.shape

# Define the number of folds
k = 5

# Create a KFold object
kf = KFold(n_splits=k)

"""### Classifier 1"""


# Initialize lists to store the results
time_to_train_lg = []
time_to_predict_lg = []
lg_accuracies = []

# Loop through the folds
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    start = time.time()
    logRegression = LogisticRegression()
    logRegression.fit(X_train, Y_train)
    end = time.time()
    time_to_train_lg.append(float(end - start))
    
    start = time.time()
    lg_predictions = logRegression.predict(X_test)
    end = time.time()
    time_to_predict_lg.append(float(end - start))
    
    lg_accuracies.append(accuracy_score(lg_predictions, Y_test))
    print("in Logistic Regression loop")

# Take the average time to train and predict across all folds
avg_time_to_train_lg = sum(time_to_train_lg) / k
avg_time_to_predict_lg = sum(time_to_predict_lg) / k
avg_lg_accuracy = sum(lg_accuracies) / k

joblib.dump(logRegression, 'bigger-classifier/saved_models/logRegression-kfolds.pkl')


print('Average Accuracy score of Logistic Regression : ', "%.3f" % avg_lg_accuracy)
print('Average Time to train Logistic Regression : ', "%.6f" % avg_time_to_train_lg, " seconds")
print('Average Time for predictions with Logistic Regression : ', "%.6f" % avg_time_to_predict_lg, " seconds")

"""### Classifier 2"""

# Initialize lists to store the results
time_to_train_svm = []
time_to_predict_svm = []
svm_accuracies = []

# Loop through the folds
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    start = time.time()
    svm = SVC()
    svm.fit(X_train, Y_train)
    end = time.time()
    time_to_train_svm.append(float(end - start))
    
    start = time.time()
    svm_predictions = svm.predict(X_test)
    end = time.time()
    time_to_predict_svm.append(float(end - start))
    
    svm_accuracies.append(accuracy_score(svm_predictions, Y_test))
    print("in svm loop")
# Take the average time to train and predict across all folds
avg_time_to_train_svm = sum(time_to_train_svm) / k
avg_time_to_predict_svm = sum(time_to_predict_svm) / k
avg_svm_accuracy = sum(svm_accuracies) / k

joblib.dump(svm, 'bigger-classifier/saved_models/svm-kfolds.pkl')


print('Average Accuracy score of SVM : ', "%.3f" % avg_svm_accuracy)
print('Average Time to train SVM : ', "%.6f" % avg_time_to_train_svm, " seconds")
print('Average Time for predictions with SVM : ', "%.6f" % avg_time_to_predict_svm, " seconds")

'''
"""### Classifier 3"""
# Initialize lists to store the results
time_to_train_bayes = []
time_to_predict_bayes = []
bayes_accuracies = []

if sparse.issparse(X):
    X = X.toarray()
# Loop through the folds
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    start = time.time()
    bayes = GaussianNB()
    bayes.fit(X_train, Y_train)
    end = time.time()
    time_to_train_bayes.append(float(end - start))

    start = time.time()
    bayes_predictions = bayes.predict(X_test)
    end = time.time()
    time_to_predict_bayes.append(float(end - start))

    bayes_accuracies.append(accuracy_score(bayes_predictions, Y_test))

# Take the average time to train and predict across all folds
avg_time_to_train_bayes = sum(time_to_train_bayes) / k
avg_time_to_predict_bayes= sum(time_to_predict_bayes) / k
avg_bayes_accuracies = sum(bayes_accuracies) / k

joblib.dump(bayes, 'saved_models/bayes-bigger.pkl')

print('Accuracy score of bayes : ', "%.3f" % avg_bayes_accuracies)
print('Time to train bayes : ', "%.6f" % avg_time_to_train_bayes, " seconds")
print('Time for predictions with bayes : ', "%.6f" % avg_time_to_predict_bayes, " seconds")
'''