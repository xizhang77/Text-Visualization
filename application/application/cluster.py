# -*- coding: utf-8 -*-
# Author: Xi Zhang. (xizhang1@cs.stonybrook.edu)

import pandas as pd
import numpy as np

import sys,os,re,glob,math,nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models, similarities

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from operator import itemgetter
from collections import Counter
from copy import deepcopy
from itertools import groupby

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

from Funcs import Accuracy


# print "Enter cluster.py"
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')
os.chdir(STATIC_DIR)
# print "This is the STATIC_DIR in cluster " + STATIC_DIR


###############################################################################
# Importing data and tokenizing the words (a-zA-Z) from each file
def import_data():
	doc_name = [] #store the name of files
	doc_context = [] #store the context in each file, one string per file
	doc_subject = []

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
		doc_subject.append(subject)
		doc_name.append(file)
		doc_context.append(out_str)

		f.close()

	return doc_name, doc_context, doc_subject

###############################################################################
# Stemming the words and Removing the stopwords for each file
def process_data(doc_name, doc_context):
	stemmer = nltk.stem.snowball.SnowballStemmer("english")
	context_stemmed = []

	for j in range(len(doc_name)):
		data_temp = doc_context[j].split(" ")
		# print data_temp
		data_stemmed = [stemmer.stem(t) for t in data_temp]
		out_str=' '.join(data_stemmed)
		# print out_str
		context_stemmed.append(out_str)

	stopwords = nltk.corpus.stopwords.words('english')
	context_clean = []

	for j in range(len(doc_name)):
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
	# vectorizer = TfidfVectorizer(min_df=0.03, stop_words='english', use_idf=True, norm='l2')	
	vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
	vectorizer.fit_transform(context)
	Matrix = vectorizer.fit_transform(context).toarray()

	Vocabulary = vectorizer.get_feature_names()
	# print len(Vocabulary)

	return Matrix, Vocabulary


###############################################################################
# SVD for dimention reduction
def SVD(Matrix):
	U, S, VT = np.linalg.svd(Matrix, full_matrices=False)
	U = U[:, 1:3]
	S = np.diag(S[1:3])
	VT = VT[1:3, :]

	return U, S, VT

###############################################################################
# Sign for the clusters
def SpectralCluster(num_clusters, Matrix):
	# LB = KMeans(n_clusters=num_clusters).fit(Matrix)
	# Matrix = np.array([[1, 2], [3, 4], [1, 1]])

	clusterlabels = np.append(np.zeros(100), np.ones(100))
	clusterlabels = np.append(clusterlabels, np.ones(100)*2)

	# AFF = np.dot(Matrix, Matrix.transpose())
	AFF = cosine_similarity(Matrix)
	D_array = AFF.sum(axis=1)
	# print D_array
	# Laplacian = D - AFF;
	InterD = np.diag(np.reciprocal(np.sqrt(D_array)))

	Laplacian = np.identity(len(AFF)) - np.dot(np.dot(InterD, AFF), InterD)
	# print Laplacian

	Eig_vals, Eig_vecs = np.linalg.eig(Laplacian)
	# print Eig_vals
	
	P = normalize(Eig_vecs[:,:num_clusters], axis=1, norm='l2')

	LB = KMeans(n_clusters=num_clusters).fit(P)

	result = LB.labels_.tolist()

	acc = Accuracy(result, clusterlabels)
	print acc

	return result, np.dot(InterD, Matrix)

###############################################################################
# Getting the keywords using submodular function
def submodular(df_kw, weight, num_keywords):
	Vocabulary = df_kw['keyword']
	# column_sum = np.amax(weight, axis=0)

	column_sum = np.sum(weight, axis = 0)
	top_index = column_sum.argsort()[-num_keywords:][::-1]
	keywords = itemgetter(*top_index)(Vocabulary)

	return keywords, top_index


###############################################################################
# Getting the keywords for the scatter plot
def getting_keywords( S, V, Vocabulary, df_doc, num_clusters, num_keywords):
	df_kw = pd.DataFrame(dict(x=V[0,:], y=V[1,:], keyword = Vocabulary)) 
	keyword_list = []
	keyword_index = []
	for i in range(num_clusters):
		subdata = df_doc[df_doc.label == i].reset_index(drop=True)
		subdata_value = subdata[['x','y']].values
		weight = np.array(np.dot(np.dot(subdata_value, S), V))

		sub_keywords, sub_index = submodular( df_kw, weight, num_keywords)

		keyword_list.append( sub_keywords )
		keyword_index.append( sub_index )

	# print keyword_list

	kw_list = [item for sublist in keyword_list for item in sublist]
	index_list = np.sort(keyword_index, axis=None)

	keyword_rank = pd.DataFrame(columns = ['keyword', 'rank', 'label'])
	keyword_rank['keyword'] = kw_list
	keyword_rank['rank'] = [20, 10, 10, 10, 10, 10, 10, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2]*3
	clusters = np.append(np.zeros(num_keywords), np.ones(num_keywords))
	clusters = np.append(clusters, np.ones(num_keywords)*2)
	keyword_rank['label'] = clusters

	keyword_show = pd.DataFrame(columns= ['x', 'y', 'keyword']) 
	keyword_show = df_kw.loc[df_kw['keyword'].isin(kw_list)]

	result = pd.merge(keyword_show, keyword_rank, on='keyword')

	return result, index_list



###############################################################################
# Generating the information needed for visualization
def scatter_layers(df_kw, df_doc, S, V):

	# Generate the 1st layer (only 3 classes are shown)
	group_layer1 = df_doc.groupby(['label']).mean() 
	counts = df_doc.groupby(['label']).size() 

	group_layer1['layer'] = counts
	group_layer1['label'] = group_layer1.index
	group_layer1 = group_layer1.reset_index(drop=True)


	group_layer1['name'] = np.nan
	group_layer1['subject'] = np.nan

	for i in range(len(group_layer1.index)):
		# Move the most important keyword to the center of the cluster
		label = group_layer1.iloc[i]['label']
		df_kw.loc[(df_kw['label'] == label) & (df_kw['rank'] == 20) , 'x'] = group_layer1.iloc[i]['x']
		df_kw.loc[(df_kw['label'] == label) & (df_kw['rank'] == 20) , 'y'] = group_layer1.iloc[i]['y']

		subdata = df_doc[df_doc.label == i].reset_index(drop=True)
		subdata = subdata[['x','y']].values

		weight = np.array(np.dot(np.dot(subdata, S), V))

		sub_keywords, sub_index = submodular( df_kw, weight, 5)
		sub_keywords = ' '.join(sub_keywords)
		group_layer1.loc[i, 'subject'] = sub_keywords
		group_layer1.loc[i, 'name'] = len(subdata)
	

	
	# Add the radius for dots in 3rd layer
	df_doc['layer'] = np.ones(len(df_doc.index))*5

	df_doc = df_doc.append(group_layer1)

	# Generate the 2nd layer (sub-clusters in each class are shown)
	for i in range(len(group_layer1.index)):
		subdata = df_doc[df_doc.label == i].reset_index(drop=True)
		subdata = subdata[['x','y']]
		subdata_value = subdata.values
		KM = KMeans(n_clusters = 6).fit(subdata_value)
		centroids = KM.cluster_centers_
		clusters = KM.labels_.tolist()
		subdata['label'] = clusters

		group_layer2 = subdata.groupby(['label']).mean()
		group_layer2 = group_layer2.reset_index(drop=True)
		group_layer2['label'] = np.ones(len(group_layer2.index)) * i
		group_layer2['layer'] = Counter(clusters).values() 
		group_layer2['name'] = np.nan
		group_layer2['subject'] = np.nan


		temp_kw = df_kw[(df_kw['label'] == i) & (df_kw['rank'] == 10)].reset_index(drop=True)

		for j in range(len(temp_kw.index)):
			pointer = temp_kw.iloc[j]['keyword']
			df_kw.loc[(df_kw['keyword'] == pointer), 'x'] = group_layer2.iloc[j]['x']
			df_kw.loc[(df_kw['keyword'] == pointer), 'y'] = group_layer2.iloc[j]['y']

			tempdata = subdata[subdata.label == j].reset_index(drop=True)
			# print i, j, tempdata
			tempvalue = tempdata[['x','y']].values
			weight = np.array(np.dot(np.dot(tempvalue, S), V))

			sub_keywords, sub_index = submodular( df_kw, weight, 5)
			sub_keywords = ' '.join(sub_keywords)
			# print sub_keywords
			group_layer2.loc[j, 'subject'] = sub_keywords
			group_layer2.loc[j, 'name'] = len(tempdata)

		# print group_layer2
		df_doc = df_doc.append(group_layer2)


	df_doc = df_doc.reset_index(drop=True)

	return df_kw, df_doc


def main():
	doc_name, doc_context, doc_subject = import_data()
	context_clean = process_data(doc_name, doc_context)
	Matrix, Vocabulary = getting_tfidf(context_clean)
	# print Matrix.shape

	num_clusters = 3
	clusters, NewMatrix = SpectralCluster(num_clusters, Matrix)

	U, S, V = SVD(NewMatrix)
	# U, S, V = SVD(Matrix)

	df_doc = pd.DataFrame(dict(x=U[:,0], y=U[:,1], label=clusters, name = doc_name, subject = doc_subject))

	num_keywords = 20
	# df_kw = getting_keywords( S, V, Vocabulary, df_doc, num_clusters, num_keywords)
	df_kw, keyword_index = getting_keywords( S, V, Vocabulary, df_doc, num_clusters, num_keywords)
	newV = V[:,keyword_index]

	df_kw, df_doc= scatter_layers(df_kw, df_doc, S, newV)

	# print df_doc
	# print df_kw

	return df_doc, df_kw


main()
