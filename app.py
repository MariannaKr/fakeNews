from flask import Flask, render_template, request
from load_models import load, predict_category, predict_categories
import pandas as pd
from pymongo import MongoClient

app = Flask(__name__)

client = MongoClient("mongodb://rootuser:rootpass@localhost:27017/")

# Select the database and collection
db = client["fakenews"]
collection = db["articles"]
collection2 = db["submit"]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    logRegression, svm, vectorizer = load()
    #news_data['content'] = news_data['author']+' '+news_data['title']#+' '+news_data['text']
    article = request.form['author']+' '+request.form['title']+' '+request.form['text']
    title = request.form['title']
    try:
        categories = predict_category(article, vectorizer, logRegression, svm, title)
    except Exception as e:
        print(e)
    return render_template('predict.html', categories=categories)

@app.route('/fetchAll', methods=['GET'])
def fetchAll():
    articles = collection.find({})
    return render_template('showAll.html', articles=articles)

@app.route('/predictCSV', methods=['POST'])
def predict_csv():
    files = request.files.getlist("article_file")
    duplicate_count = 0
    predictions = []
    for file in files:
        data = pd.read_csv(file)
        logRegression, svm, vectorizer = load()
        for index, row in data.iterrows():
            title = str(row['title'])
            article_id = row['id']
            item = collection2.find_one({'_id': article_id}, {'label': 1})
            label = item['label']
            text = ''
            text += str(row['author']) + ' '
            text += str(row['title']) + ' '
            #text += str(row['text'])
            pred = predict_categories(text, vectorizer, logRegression, svm, title, article_id, label)
            predictions.append(pred)
            predictionLog = str(pred[1])
            predictionSVM = str(pred[2])
            document = {"_id": article_id, "title": title, "author": (row['author']), "text": row['text'], "predictionLog": predictionLog, "predictionSVM": predictionSVM, "label": label}
            if collection.find_one({"_id": article_id}):
                duplicate_count += 1
                continue
            collection.insert_one(document)
        if duplicate_count > 0:
            print(f"{duplicate_count} articles already in db")
    return render_template('predictMany.html', categories=predictions)

@app.route('/read/<string:id>')
def read(id):
    article_id = int(id)
    article = collection.find_one({'_id': article_id})
    return render_template('read.html', article=article)

@app.route('/statistics', methods=['GET'])
def statistics():
    total_articles = collection.count_documents({})
    log_match_1 = collection.count_documents({"predictionLog": {"$eq": "1"}, "label": {"$eq": "1"}})
    log_match_2 = collection.count_documents({"predictionLog": {"$eq": "0"}, "label": {"$eq": "0"}})
    log_match = log_match_1 + log_match_2
    print(log_match_1)
    svm_match_1 = collection.count_documents({"predictionSVM": {"$eq": "1"}, "label": {"$eq": "1"}})
    svm_match_2 = collection.count_documents({"predictionSVM": {"$eq": "0"}, "label": {"$eq": "0"}})
    svm_match = svm_match_1 + svm_match_2
    accuracy_log = log_match / total_articles
    accuracy_svm = svm_match / total_articles
    return render_template('statistics.html', total_articles=total_articles, log_match=log_match, svm_match=svm_match, accuracy_log=accuracy_log, accuracy_svm=accuracy_svm)


if __name__ == '__main__':
    app.run(debug=True)
