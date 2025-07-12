#environment: textgcn
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str, loadWord2Vec
import sys
import pandas as pd
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np
#import random

if len(sys.argv) != 2:
	sys.exit("Use: python remove_words_rad_finding.py <dataset>")

datasets = ['rad']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

#read train set
train_list = open("./Finding/Train/train.txt",'r')
train_data = []; train_label = []
for line in train_list:
    label, text = line.split('\t')
    train_data.append(text)
    if label == 'followup':
        train_label.append(1)
    elif label == 'nofollowup':
        train_label.append(0)
train_list.close()


#read validation set
validation_list = open("./Finding/Validation/validation.txt",'r')
validation_data = []; validation_label = []
for line in validation_list:
    label, text = line.split('\t')
    validation_data.append(text)
    if label == 'followup':
        validation_label.append(1)
    elif label == 'nofollowup':
        validation_label.append(0)
validation_list.close()
#df_validation = pd.DataFrame(validation_data, columns = ['Report Text'])

#read test set
test_list = open("./Finding/Test/test.txt",'r')
test_data = []; test_label = []
for line in test_list:
    label, text = line.split('\t')
    test_data.append(text)
    if label == 'followup':
        test_label.append(1)
    elif label == 'nofollowup':
        test_label.append(0)
test_list.close()


word_freq = {}  # to remove rare words

for doc_content in train_data:
    #temp = clean_str(doc_content)
    words = doc_content.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

#remove low frequence words from the documentary
clean_docs_train = []
for doc_content_train in train_data:
    words = doc_content_train.split()
    doc_words = []
    for word in words:
        if word in word_freq and word_freq[word] >= 5:
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip()
    clean_docs_train.append(doc_str)

temp = list(zip(clean_docs_train, train_label))
np.random.shuffle(temp)
clean_docs_train, train_label = zip(*temp)
clean_docs_train = list(clean_docs_train); train_label = np.array(train_label)
print(len(clean_docs_train)); print(len(train_label));

np.save('./Finding/text_gcn/train.label', train_label)

clean_corpus_str_train = '\n'.join(clean_docs_train) #use \n to connect each document.

#save cleaned corpus to file
f = open('./Finding/text_gcn/train.clean.txt', 'w')
#f = open('data/wiki_long_abstracts_en_text.clean.txt', 'w')
f.write(clean_corpus_str_train)
f.close()


#remove low frequence words from the documentary
clean_docs_validation = []
for doc_content_validation in validation_data:
    #temp = clean_str(doc_content)
    words = doc_content_validation.split()
    doc_words = []
    for word in words:
        if word in word_freq and word_freq[word] >= 5:
            doc_words.append(word)
        #else:
        #    print(word)
    doc_str = ' '.join(doc_words).strip()
    clean_docs_validation.append(doc_str)


clean_docs_validation = list(clean_docs_validation); validation_label = np.array(validation_label)
print(len(clean_docs_validation)); print(len(validation_label));
np.save('./Finding/text_gcn/validation.label', validation_label)

clean_corpus_str_validation = '\n'.join(clean_docs_validation) #use \n to connect each document.

#save cleaned corpus to file
f = open('./Finding/text_gcn/validation.clean.txt', 'w')
#f = open('data/wiki_long_abstracts_en_text.clean.txt', 'w')
f.write(clean_corpus_str_validation)
f.close()

#remove low frequence words from the documentary
clean_docs_test = []
for doc_content_test in test_data:
    #temp = clean_str(doc_content)
    words = doc_content_test.split()
    doc_words = []
    for word in words:
        if word in word_freq and word_freq[word] >= 5:
            doc_words.append(word)
        #else:
        #    print(word)
    doc_str = ' '.join(doc_words).strip()
    clean_docs_test.append(doc_str)


clean_docs_test = list(clean_docs_test); test_label = np.array(test_label)
print(len(clean_docs_test)); print(len(test_label));
np.save('./Finding/text_gcn/test.label', test_label)

clean_corpus_str_test = '\n'.join(clean_docs_test) #use \n to connect each document.

#save cleaned corpus to file
f = open('./Finding/text_gcn/test.clean.txt', 'w')
f.write(clean_corpus_str_test)
f.close()
