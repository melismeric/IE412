#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import io
import pandas as pd
import numpy as np


# In[2]:


path = r"C:\Users\melis.meric.TY\Desktop\Sari_Meric_Ie413_Asgn2\DocumentClassification"
documents = {'mis':['MIS'],'phil':['PHIL']}


# In[4]:


def readdata(fullpath):
    for root, subfolders, filelist in os.walk(fullpath, topdown=True):
        for currfile in filelist:
            filepath = os.path.join(root, currfile)
            f = io.open(filepath, encoding="latin-1")
            content = f.read()
            # text preprocessing
            #...
            yield filepath, content
def createdataframe(fullpath, label):
    rows = []
    index = []
    for fname, fcontent in readdata(fullpath):
        rows.append({'content':fcontent, 'label':label})
        index.append(fname)
    
    df = pd.DataFrame(rows, index=index)
    return df
    
dataEmails = pd.DataFrame({'content':[], 'label': []})
for label, foldernames in documents.items():
    for foldername in foldernames:
        documentPath = os.path.join(path, foldername)
        dftemp = createdataframe(documentPath, label)
        dataEmails = dataEmails.append(dftemp)


# In[5]:


dataEmails['content'][0:10], dataEmails['label'][0:10]


# In[24]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text

stopFolder = open("StopwordsDict.txt","r")
#print(stopFolder.read())
my_stop_words = text.ENGLISH_STOP_WORDS.union(["(3+1+0)"]).union(["(3+2+0)"])
#print(my_stop_words)


# In[26]:


cvectorizer = CountVectorizer(ngram_range=(1,1), min_df = 1, max_features=None) 
vectorizerTFIDF = TfidfVectorizer(ngram_range=(1,1), stop_words=my_stop_words, min_df = 1, max_features=None)
vectorizerTFIDFubgrams = TfidfVectorizer(ngram_range=(1,1), stop_words=my_stop_words, min_df = 1, max_features=None)


# In[27]:


y = dataEmails["label"]
type(y)
print(y.value_counts())


# In[28]:


from sklearn.model_selection import KFold, StratifiedKFold
def evaluatemodelsbow(alldocs, y, classifiers=[], k=10, rand_state=42):
    clfscores = dict()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=rand_state)
    for clf in classifiers:
        scores = []
        for train_index, test_index in skf.split(alldocs,y):
            X_train_docs, X_test_docs, y_train, y_test = alldocs[train_index], alldocs[test_index], y[train_index], y[test_index]
            # Create term by document matrix with vectorizers
            
            # following two lines does the same as the third one
            # X_train = vectorizerTFIDF.fit(X_train_docs)
            # X_train = vectorizerTFIDF.transform(X_train_docs)
            X_train = vectorizerTFIDF.fit_transform(X_train_docs)
            X_test = vectorizerTFIDF.transform(X_test_docs)
            
            #X_train_tfidf_ubg = vectorizerTFIDFubgrams.fit_transform(X_train) # used for Unigram+Bigram features with TfIdf
            #X_train_count = cvectorizer.fit_transform(X_train) # used for Unigram features with Counts
            
            # Train classifier
            clf.fit(X_train, y_train)
            # Predict test labels
            y_pred = clf.predict(X_test)
            # Compute the accuracy scores
            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)
        clfscores[clf] = scores
    return clfscores


# In[29]:


# Classification models - create instances and then create a list of them to feed into evaluatemodelsbow function
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
logreg = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
mlp = MLPClassifier()
classifiers = [logreg, dt, rf, mlp]
alldocs = dataEmails["content"]
y = dataEmails['label']
kfold = 5
scores = evaluatemodelsbow(alldocs, y, classifiers, k=kfold)
print(scores)


# In[31]:


# UNKNOWN LABELING
mlp = MLPClassifier() # best classifier according to k-fold CV
tbdmatrix = vectorizerTFIDF.fit_transform(alldocs)
mlp.fit(tbdmatrix, y)
# read UNLABELED DOCS into unknowndocs
dataUnknownEmails = pd.DataFrame({'content':[], 'label': []})
folderpath4Unknowns =  r"C:\Users\melis.meric.TY\Desktop\Sari_Meric_Ie413_Asgn2\DocumentClassification\UNLABELED"
unknowndocs = createdataframe(folderpath4Unknowns, "UNKNOWN")
# transform unlabeled documents to get term by doc matrix
tbdUnk = vectorizerTFIDF.transform(unknowndocs['content'])
# predict the labels
unkLabels = mlp.predict(tbdUnk)
print(unkLabels)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




