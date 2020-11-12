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

rna_junctions = pd.read_table('insert_file_name_here.txt')