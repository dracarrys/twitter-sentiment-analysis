import pandas as pd
import numpy as np
import nltk
#nltk.download('punkt')

from sklearn.model_selection import cross_val_score
from sklearn import feature_extraction
from sklearn import svm


names=['label','id', 'date', 'q', 'u', 'text']
#import the datasets
df = pd.read_csv("trainingandtestdata/train.csv",header=None)
df_dev = pd.read_csv("trainingandtestdata/test.csv", header=None)

df_all=pd.concat([df[:10000], df_dev])
df_all.columns = names
df_all = df_all[['label', 'text']]



scores_array = []
for i in range(1):
    i += 1
    # create the new columns words with count vectorizer
    count_vectorizer = feature_extraction.text.CountVectorizer(
        lowercase=True, # for demonstration, True by default
        tokenizer=nltk.word_tokenize, # use the NLTK tokenizer
        stop_words='english', # remove stop words
        min_df=1, # minimum document frequency, i.e. the word must appear more than once.
        ngram_range=(i, i))


    #transform the values in columns with tf-idf algorithm
    processed_corpus = count_vectorizer.fit_transform(df_all['text'])
    processed_corpus = feature_extraction.text.TfidfTransformer().fit_transform(processed_corpus)


    #apply svm
    clf = svm.SVC(kernel='linear', C=1, random_state=42)
    scores = cross_val_score(clf, processed_corpus, df_all['label'], cv=5, scoring='accuracy')
    print(scores)
    scores_array.append(max(scores))
