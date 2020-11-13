'''
@author: Benfeard Williams
Created on Tue Nov 3, 2020

Purpose: create a model that can predict the alignment of stems in an RNA 3-way junction
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split

rna_3way = pd.read_csv('rna_junctions.csv')
#print(rna_3way.head())

# create a mapping from rna label to junction type
lookup_junction_type = dict(zip(rna_3way.rna_label.unique(), rna_3way.junction_type.unique()))
#print(lookup_junction_type)

# split dataset
X = rna_3way[['stem_a', 'junction_a', 'stem_b', 'junction_b', 'stem_c', 'junction_c']]
y = rna_3way['rna_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# naive Bayes classifer
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(X_train, y_train)

classifier.score(X_test, y_test)
predicted = classifier.predict(X_test)
