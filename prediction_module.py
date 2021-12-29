import pickle

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = stopwords.words("english")
stop_words.extend(stopwords.words("russian"))
stop_words = set(stop_words)


def tokenise(tr, istrain=True):
    tr['Text'] = tr.Text.apply(str.lower)
    tr['Tokens'] = tr.Text.apply(word_tokenize)

    def to_alpha(tokens):
        words = [word for word in tokens if word.isalpha()]
        return words

    tr['alpha_tokens'] = tr.Tokens.apply(to_alpha)

    def remove_stopwords(tokens):
        words_s = [word for word in tokens if not word in stop_words]
        return words_s

    tr['alpha_stop_tokens'] = tr.alpha_tokens.apply(remove_stopwords)

    if istrain:
        data = tr[['alpha_stop_tokens', 'Score']]
    else:
        data = tr[['alpha_stop_tokens']]

    def join_list(tab):
        return " ".join(tab)

    # data['alpha_stop_tokens'] = data['alpha_stop_tokens'].apply(join_list)

    return data


def prepare_data(data):
    return tokenise(data, istrain=False)


def make_vec(X, num_top):
    matrix = np.zeros((len(X), num_top))
    for i, row in enumerate(X):
        matrix[i, list(map(lambda tup: tup[0], row))] = list(map(lambda tup: tup[1], row))
    return matrix


def transform(df, tf_model, model):
    with open('mydict.pkl', 'rb') as file:
        mydict = pickle.load(file)
    corpus = [mydict.doc2bow(text) for text in df]
    corpus = tf_model[corpus]
    corpus = model[corpus]
    corpus = make_vec(corpus, model.num_topics)
    return corpus
