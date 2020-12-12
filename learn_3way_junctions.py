'''
@author: Benfeard Williams
Created on Tue Nov 3, 2020

Purpose: create a model that can predict the alignment of stems in an RNA 3-way junction
'''
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys

rna_3way = pd.read_csv('rna_junctions.csv')
#print(rna_3way.head())

# create a mapping from rna label to junction type
lookup_junction_type = dict(zip(rna_3way.rna_label.unique(), rna_3way.junction_type.unique()))
#print(lookup_junction_type)

# dealing with RNA features being sequences of nucleic acids
seq_text = rna_3way['stem_a']#[['stem_a', 'junction_a', 'stem_b', 'junction_b', 'stem_c', 'junction_c']]
vectorizer = CountVectorizer()
vectorizer.fit(seq_text)
vector = vectorizer.transform(seq_text)

# look at results
print(vector.toarray())
print(vectorizer.vocabulary_)

# I need to make all my arrays the same length for classification
def padarray64(my_array):
    t = 64 - len(my_array)
    return np.pad(my_array, pad_width=(0, t), mode='constant')
    
# Pad each sequence with Z so all the stems and junctions are 64 char
def padseq64(my_seq):
    x = my_seq.ljust(64, "Z")
    #print(len(x))
    return x


# Ordinal encoding of RNA sequence data
# "GUCA" --> [0.25, 0.5, 0.75, 1.0] 
# although I hope this doesn't weight actual nucleotides
# Let's make zero lenth junctions = 0
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['G', 'U', 'C', 'A', 'Z']))


def ordinal_encoder(my_seq):
    my_seq = np.array(list(my_seq))
    integer_encoded = label_encoder.transform(my_seq)
    
    float_encoded = integer_encoded.astype(float)
    
    float_encoded[float_encoded == 0] = 0.25 # A
    float_encoded[float_encoded == 1] = 0.50 # C
    float_encoded[float_encoded == 2] = 0.75 # G
    float_encoded[float_encoded == 3] = 1.00 # U
    float_encoded[float_encoded == 4] = 0.00 # Z (empty junction)
    
    return float_encoded


# Optional one-hot encoding for future deep learning method
# "ACGUZ" --> [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0]
def one_hot_encoder(my_seq):
    my_seq = np.array(list(my_seq), dtype=np.float)
    integer_encoded = label_encoder.transform(my_seq)
    
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int, categories=[range(5)])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded = np.delete(onehot_encoded, -1, 1)
    
    return onehot_encoded
     
# non label columns
columns = [column for column in rna_3way][3:]

for column in columns:
    rna_3way[column] = rna_3way[column].apply(padseq64)    
    
rna_3way['final'] = (
                        rna_3way['stem_a'] + rna_3way['junction_a'] + 
                        rna_3way['stem_b'] + rna_3way['junction_b'] + 
                        rna_3way['stem_c'] + rna_3way['junction_c']
                    )
                    
rna_3way['final'] = rna_3way['final'].apply(ordinal_encoder)
#for column in columns:
    #rna_3way[column] = rna_3way[column].apply(ordinal_encoder)
    #rna_3way[column] = rna_3way[column].apply(one_hot_encoder) #don't use this one
    
    # make vectors the same length
    #rna_3way[column] = rna_3way[column].apply(padarray64)
 
## new problem is that none of the sequences are the same length

#rna_3way['final'] = pd.concat(rna_3way['stem_a'], rna_3way['junction_a'])
#print(rna_3way['final'].shape)
#print(rna_3way['final'])

my_matrix = pd.DataFrame(rna_3way['final'].values.tolist())
print(my_matrix.shape)


# make a 1-dimensional view of arr
#arr = np.array(rna_3way[['stem_a', 'junction_a', 'stem_b', 'junction_b', 'stem_c', 'junction_c']])
#print(arr.shape)

'''
i'll need to pad each sequence with Z's
then add them together rna_3way[final]  = 'stem_a" + stem_b + ...
then encode them
'''

# split dataset
# blue = A, green = B, red = C
X = my_matrix #rna_3way['final']
y = rna_3way['rna_label']
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
 
# feature scaling for training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# neural network   
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)

print("i trained the neural network")

predictions = mlp.predict(X_test)
print(predictions)

# evaluate the algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# naive Bayes classifer
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(X_train, y_train)

classifier.score(X_test, y_test)
predicted = classifier.predict(X_test)
print(predicted)

print("i trained the multinomial network")


