# Hey, it's me @alviankosim

import json
import os
import random
import pandas as pd

from flask import Flask, render_template, request
from utils.filecl import preprocess
from utils.kw import (
    kw_textrank, kw_keybert
)
from utils.tfidf import (
    tf, idf_doc, tf_idf, cos_sim
)
app = Flask(__name__,
            static_url_path='', 
            static_folder='public')

data = pd.read_csv('storage/scopus1.csv')
print("Data Loaded")

# home route
@app.route("/")
def home():
    return render_template("home.html")

# search route
@app.route("/search", methods = ['GET'])
def search():
    q = request.args.get('q')
    return data[data['Title'].str.contains(q, case=False)].iloc[:10][['Abstract', 'Title']].to_json()

# result route
@app.route("/result", methods = ['GET'])
def result():
    id = int(request.args.get('id'))
    q = request.args.get('q')
    
    # one
    one_data = data.iloc[id][['Abstract', 'Title']]
    cleaned_one_data = preprocess(one_data['Abstract'], True)
    idfed_one = idf_doc([cleaned_one_data])
    tf_normalized_one = tf(cleaned_one_data)
    tf_x_idf_one = tf_idf(cleaned_one_data, tf_normalized_one, idfed_one)

    # five
    five_data = data[~data.index.isin([id])][data['Title'].str.contains(q, case=False)].sample(n=5)[['Abstract', 'Title']]
    cleaned_five_data = five_data['Abstract'].apply(preprocess)
    idfed = idf_doc(cleaned_five_data)

    # tf idf for the five
    result = []
    for idx, text in enumerate(cleaned_five_data):
        # normalized tf
        tf_normalized = tf(text)

        # tf * idf
        tf_x_idf = tf_idf(cleaned_one_data, tf_normalized, idfed)

        # cosine similarity
        cos_sim_ = cos_sim(tf_x_idf_one, tf_x_idf)

        # unstemmed data for searching keywords
        data_unstemmed = preprocess(five_data['Abstract'].iloc[idx], False)

        result.append({
            'title': five_data['Title'].iloc[idx],
            'textrank': kw_textrank(data_unstemmed),
            'keybert' : kw_keybert(data_unstemmed),
            'cos_sim_': cos_sim_
        })
    
    return json.dumps({
        'one_data': one_data['Title'],
        'result': sorted(result, key=lambda i: i['cos_sim_'], reverse=True)
    })

if __name__ == "__main__":
    app.run(port=3600, host="0.0.0.0", debug=True)