from flask import Flask, render_template, request
from load_models import load, predict_category
import pandas as pd

app = Flask(__name__)

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
            text = ''
            if not pd.isna(row['author']):
                text += str(row['author']) + ' '
            if not pd.isna(row['title']):
                text += str(row['title']) + ' '
            text += str(row['text'])
            pred = predict_category(text, vectorizer, logRegression, svm, title)
            predictions.append(pred)
    return render_template('predictMany.html', categories=predictions)

if __name__ == '__main__':
    app.run(debug=True)

