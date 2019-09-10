from application import app
from application import cluster

from flask import render_template
import unicodedata
import os.path
import pandas as pd


# print "Enter views.py"
# print os.getcwd()

@app.route('/')
@app.route('/', methods=['POST'])
def my_form_post():
	STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')
	# print "This is STATIC_DIR in views" + STATIC_DIR

	df_doc, df_kw = cluster.main()
	doc_data = df_doc.to_json(orient='records')
	kw_data = df_kw.to_json(orient='records')

	return render_template("post.html",
		static_folder=STATIC_DIR,
		doc_data = doc_data,
		kw_data = kw_data)
