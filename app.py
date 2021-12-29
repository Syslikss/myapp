import json
import pickle

import nltk
import pandas as pd
from flask import Flask, request
from gensim.corpora.dictionary import Dictionary
from gensim.models import TfidfModel
from nltk.corpus import stopwords

from prediction_module import prepare_data, transform

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words("english") + stopwords.words("russian"))


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print('====')
        data = json.loads(request.data)
        data = pd.DataFrame(data)
        data = prepare_data(data)
        mydict = Dictionary(data.alpha_stop_tokens)
        corpus = [mydict.doc2bow(text) for text in data.alpha_stop_tokens]
        tf_model = TfidfModel(corpus)
        vectorized_data = transform(data.alpha_stop_tokens, tf_model, vectorizer)

        data['Prediction'] = model.predict(vectorized_data)

        return data[['alpha_stop_tokens', 'Prediction']].to_json()


if __name__ == '__main__':
    app.run()
