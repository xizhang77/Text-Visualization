# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from flask import Flask, request, jsonify
from flask import render_template
import json

import random
import unicodedata
import os.path

app = Flask(__name__)

@app.route('/')
# def my_form():
#     return render_template("index.html")

@app.route('/', methods=['POST'])
def my_form_post():
    STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/')

    df = pd.read_csv(STATIC_DIR + 'documents.csv')
    doc_data = df.to_json(orient='records')

    df = pd.read_csv(STATIC_DIR + 'keyword.csv')
    kw_data = df.to_json(orient='records')

    return render_template("post.html",
        static_folder=STATIC_DIR,
        doc_data = doc_data,
        kw_data = kw_data)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)