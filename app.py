from flask import Flask, render_template, request
from load_models import load, predict_category
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
    article = request.form['article']+' '+request.form['title']+' '+request.form['author']
    title = request.form['title']
    try:
        categories = predict_category(article, vectorizer, logRegression, svm,title)
    except Exception as e:
        print(e)
    return render_template('predict.html', categories=categories)

@app.route('/predictCSV', methods=['POST'])
def predict_csv():
    files = request.files.getlist("article_file")
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
            if not pd.isna(row['author']):
                text += str(row['author']) + ' '
            if not pd.isna(row['title']):
                text += str(row['title']) + ' '
            text += str(row['text'])
            pred = predict_category(text, vectorizer, logRegression, svm, title, article_id, label)
            predictions.append(pred)
            predictionLog = str(pred[1])
            predictionSVM = str(pred[2])
            document = {"_id": article_id, "title": title, "author": (row['author']), "text": row['text'], "predictionLog": predictionLog, "predictionSVM": predictionSVM, "label": label}
            collection.insert_one(document)
    return render_template('predictMany.html', categories=predictions)

if __name__ == '__main__':
    app.run(debug=True)

