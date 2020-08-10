# Python3 code for preprocessing text 
import nltk
import string
import re
import heapq
import pandas as pd
import numpy as np 

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

NUM_WORDS = 100

def tokenize_input(input_df):

	dataset = []
	y = input_df['Classification'].values

	stopset = set(stopwords.words('english'))

	# preprocessing
	for clause in input_df['Clause Text'].values: # this ordering has to match 'y' label ordering
		clause = clause.lower()
		clause = re.sub(r'\W', ' ', clause) # remove nonwords
		clause = re.sub(r'\s+', ' ', clause) # remove punctuation
		dataset.append(clause)

	# get word to counts
	word_to_count = {} 
	for clause in dataset:
		words = nltk.word_tokenize(clause)
		for word in words:
			if word in stopset:
				continue
			elif word not in word_to_count.keys(): 
				word_to_count[word] = 1
			else:
				word_to_count[word] += 1

	freq_words = heapq.nlargest(NUM_WORDS, word_to_count, key=word_to_count.get)

	X = []
	for clause in dataset:
	    vector = [] 
	    for word in freq_words: 
	        if word in nltk.word_tokenize(clause): 
	            vector.append(1) 
	        else: 
	            vector.append(0) 
	    X.append(vector)
	X = np.asarray(X)

	return X


def tokenize_input_alt(input_df):

	token = RegexpTokenizer(r'[a-zA-Z0-9]+')
	cv = CountVectorizer(lowercase=True,stop_words='english', ngram_range = (1,1), tokenizer = token.tokenize)
	X = cv.fit_transform(input_df['Clause Text'])

	return X


def main():

	df = pd.read_csv('/Users/parthpendurkar/Desktop/raw.csv')
	
	X = tokenize_input_alt(df)
	y = df['Classification']

	# train-test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

	model = MultinomialNB().fit(X_train, y_train)
	predicted= model.predict(X_test)
	
	print("acc: ", metrics.accuracy_score(y_test, predicted))
	print("f1 score: ", metrics.f1_score(y_test, predicted))

	tn, fp, fn, tp = metrics.confusion_matrix(y_test, predicted).ravel()

	print(tn / len(predicted))
	print(fp / len(predicted))
	print(fn / len(predicted))
	print(tp / len(predicted))


if __name__ == '__main__':
	main()