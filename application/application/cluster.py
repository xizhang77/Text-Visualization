# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import nltk
import codecs
import os
import re
import sys
import collections
import glob
import math

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from gensim import corpora, models, similarities
from scipy.stats import gaussian_kde
from sklearn.neighbors.kde import KernelDensity

from operator import itemgetter
from collections import Counter

# print "Enter cluster.py"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')
os.chdir(STATIC_DIR)
# print "This is the STATIC_DIR in cluster " + STATIC_DIR
###############################################################################
# Importing data and tokenizing the words (a-zA-Z) from each file
def import_data():
	files_name = [] #store the name of files
	context_str = [] #store the context in each file, one string per file
	file_subject = []

	for file in glob.glob("*"): 
		f = open(file, 'r')	
		lines_list = []
		for line in f.readlines():
			# Get the subject from each file (For visualization only)
			if 'Subject' in line:
				subject = re.findall(r'[a-zA-Z]+', line)
				subject = ' '.join(subject)
				subject = subject.split(' ', 1)[1]
			match = re.findall(r'[a-zA-Z]+', line) #[\w]: matches any alphanumeric character and the underscore; equivalent to [a-zA-Z0-9_].
			lines_list = np.append(lines_list, match, axis = 0)
		out_str=' '.join(lines_list)
		file_subject.append(subject)
		files_name.append(file)
		context_str.append(out_str)

		f.close()

	return files_name, context_str, file_subject

def document_filter(files_name, context_str, string):
	sub_file = []
	for j in range(len(files_name)):
		data_temp = context_str[j].split(" ")
		# print data_temp
		if string in data_temp:
			print files_name[j]
			sub_file.append(files_name[j])
	return sub_file

###############################################################################
# Stemming the words and Removing the stopwords for each file
def process_data(files_name, context_str):
	stemmer = SnowballStemmer("english")
	context_stemmed = []

	for j in range(len(files_name)):
		data_temp = context_str[j].split(" ")
		# print data_temp
		data_stemmed = [stemmer.stem(t) for t in data_temp]
		out_str=' '.join(data_stemmed)
		# print out_str
		context_stemmed.append(out_str)

	stopwords = nltk.corpus.stopwords.words('english')
	context_clean = []

	for j in range(len(files_name)):
		data_temp = context_stemmed[j].split(" ")
		# print data_temp
		texts = [word for word in data_temp if word not in stopwords]
		out_str=' '.join(texts)
		# print texts
		context_clean.append(out_str)

	return context_clean

###############################################################################
# Getting the Tf-IDF matrix
def getting_tfidf(context):
	vectorizer = TfidfVectorizer(min_df=0.05, stop_words='english', use_idf=True)
	vectorizer.fit_transform(context)
	Matrix = vectorizer.fit_transform(context).toarray()

	Vocabulary = vectorizer.get_feature_names()

	return Matrix, Vocabulary

###############################################################################
# Sign for the clusters

def clustering(num_clusters, Matrix):
	KM = KMeans(n_clusters=num_clusters).fit(Matrix)
	clusters = KM.labels_.tolist()

	clusters = np.append(np.zeros(100), np.ones(100))
	clusters = np.append(clusters, np.ones(100)*2)

	return clusters

###############################################################################
# SVD for dimention reduction
def SVD(Matrix):
	U, S, V = np.linalg.svd(Matrix, full_matrices=False)
	U = U[:,1:3]
	S = np.identity(2)*S[1:3]
	V = V[1:3,:]

	return U, S, V


###############################################################################
# Getting the coordinates for the most important keywords
def getting_keywords( S, V, Vocabulary, df_doc, num_clusters, num_keywords):
	df_kw = pd.DataFrame(dict(x=V[0,:], y=V[1,:], keyword = Vocabulary)) 

	keyword_list = []
	for i in range(num_clusters):
		subdata = df_doc[df_doc.label == i]
		subdata = subdata.reset_index()
		sub_rank = []
		# Get the top 100 keywords in each document
		for j in range(len(subdata)):
			doc_index =np.array([ subdata.ix[j]['x'] , subdata.ix[j]['y'] ])
			weight = np.array(np.dot(np.dot(doc_index, S), V))
			rank = weight.argsort()[-100:][::-1]
			sub_rank.extend(rank)
		# Get the most common keywords in each cluster (100 documents/cluster)
		top_index= [word for word, word_count in Counter(sub_rank).most_common(num_keywords)]
		most_common_words = itemgetter(*top_index)(Vocabulary)
		print most_common_words
		keyword_list.append( most_common_words )

	flat_list = [item for sublist in keyword_list for item in sublist]
	print flat_list

	keyword_rank = pd.DataFrame(columns = ['keyword', 'rank'])
	keyword_rank['keyword'] = flat_list
	keyword_rank['rank'] = [20, 10, 10, 10, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2]*3
	keyword_show = pd.DataFrame(columns= ['x', 'y', 'keyword']) 
	keyword_show = df_kw.loc[df_kw['keyword'].isin(flat_list)]

	result = pd.merge(keyword_show, keyword_rank, on='keyword')

	return result


def main1():
	files_name, context_str, file_subject = import_data()
	context_clean = process_data(files_name, context_str)
	Matrix, Vocabulary = getting_tfidf(context_clean)

	num_clusters = 3
	clusters = clustering(num_clusters, Matrix)
	U, S, V = SVD(Matrix)

	df_doc = pd.DataFrame(dict(x=U[:,0], y=U[:,1], label=clusters, name = files_name, subject = file_subject))

	num_keywords = 20
	df_kw = getting_keywords( S, V, Vocabulary, df_doc, num_clusters, num_keywords)

	return df_doc, df_kw

def main2(string):
	files_name, context_str, file_subject = import_data()
	context_clean = process_data(files_name, context_str)
	Matrix, Vocabulary = getting_tfidf(context_clean)

	num_clusters = 3
	clusters = clustering(num_clusters, Matrix)
	U, S, V = SVD(Matrix)

	df_doc = pd.DataFrame(dict(x=U[:,0], y=U[:,1], label=clusters, name = files_name, subject = file_subject))

	sub_file = document_filter(files_name, context_clean, string)
	sub_doc = df_doc[df_doc.name.isin(sub_file)]
	sub_doc = sub_doc,reset_index(drop = True)

	num_keywords = 20
	df_kw = getting_keywords( S, V, Vocabulary, df_doc, num_clusters, num_keywords)

	return sub_doc
