import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
import struct
from nltk.corpus import stopwords
from nltk import word_tokenize
import xlrd
import random
import numpy as np
import csv
import math
import pickle
import collections


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
        else:
            word_embeddings_dim = int(row[1])
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map, word_embeddings_dim

#option = [mean, max, maxmin]
def get_doc_vec_options(dfset, word_vector_map, word_embeddings_dim, option):

    real_size = len(dfset)
    data_all_word_vec = []
    for i in range(0, real_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        #initial doc vector to get min
        doc_vec_min = np.array([math.inf for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        # initial doc vector to get max
        doc_vec_max = np.array([-math.inf for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        doc_words = dfset['Report Text'][i]  # one document
        if type(doc_words) is float:
            print(doc_words)
            doc_words = str(doc_words)
        doc_words = doc_words.replace('.', ' ')
        doc_words = doc_words.replace(',', ' ')
        words = word_tokenize(doc_words)  # all words in one document
        wordcount = 0
        for word in words:
            if word in word_vector_map:
                wordcount = wordcount + 1
                #contain numerical positive and negative values
                word_vector = word_vector_map[word]
                # print(doc_vec)
                # print(np.array(word_vector))
                if option == 'mean':
                    doc_vec = doc_vec + np.array(word_vector)  # add all word vector to represent doc vec
                elif option == 'max':
                    doc_vec_max = np.maximum(doc_vec_max, word_vector)
                elif option == 'min':
                    doc_vec_min = np.minimum(doc_vec_min, word_vector)
                elif option == 'maxmin':
                    doc_vec_max = np.maximum(doc_vec_max, word_vector)
                    doc_vec_min = np.minimum(doc_vec_min, word_vector)
        if wordcount != 0:
            if option == 'mean':
                doc_vec = doc_vec / wordcount
                data_all_word_vec.append(doc_vec)
            elif option == 'max':
                doc_vec = doc_vec + doc_vec_max
                data_all_word_vec.append(doc_vec)
            elif option == 'min':
                doc_vec = doc_vec + doc_vec_min
                data_all_word_vec.append(doc_vec)
            elif option == 'maxmin':
                doc_vec = np.hstack((doc_vec, doc_vec)) + np.hstack((doc_vec_max, doc_vec_min))
                data_all_word_vec.append(doc_vec)
        else: # delete the null samples
            data_all_word_vec.append(doc_vec)
            #raise Exception('Impression part is null')
            #print("Impression part is null, delete the null samples")
            #print(i)
            #print('\n')
            #dfset = dfset.drop(i)
    dfset = dfset.reset_index(drop=True)    #reset the index (https://pandas.pydata.org/pandas-docs/version/0.22.0/generated/pandas.DataFrame.reset_index.html)
    dfset['doc vec'] = data_all_word_vec

    return  dfset
