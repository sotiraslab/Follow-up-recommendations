import os
from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from dataset import Word2VecDataset
from word2vec import Word2VecModel, WordVectors
from utils import  loadWord2Vec, get_doc_vec_options
import argparse
import pickle

parser = argparse.ArgumentParser(description="Parse command-line arguments")
parser.add_argument('--operation', type=str, required=True, help="Mean/max/min operation to word vectors")

args = parser.parse_args()

#read train set
train_list = open("./DocSVM/Finding/Train/train.txt",'r')
train_data = []; train_label = []
for line in train_list:
    label, text = line.split('\t')
    train_data.append(text)
    if label == 'followup':
        train_label.append(1)
    elif label == 'nofollowup':
        train_label.append(0)
train_list.close()
df_train = pd.DataFrame(train_data, columns = ['Report Text'])

#read validation set
validation_list = open("./DocSVM/Finding/Validation/validation.txt",'r')
validation_data = []; validation_label = []
for line in validation_list:
    label, text = line.split('\t')
    validation_data.append(text)
    if label == 'followup':
        validation_label.append(1)
    elif label == 'nofollowup':
        validation_label.append(0)
validation_list.close()
df_validation = pd.DataFrame(validation_data, columns = ['Report Text'])

#read test set
test_list = open("./DocSVM/Finding/Test/test.txt",'r')
test_data = []; test_label = []
for line in test_list:
    label, text = line.split('\t')
    test_data.append(text)
    if label == 'followup':
        test_label.append(1)
    elif label == 'nofollowup':
        test_label.append(0)
test_list.close()
df_test = pd.DataFrame(test_data, columns = ['Report Text'])

#Read Word Vectors
word_vector_file = './DocSVM/files/output/300/finding/model'
vocab_, embd, word_vector_map, word_embeddings_dim = loadWord2Vec(word_vector_file)
del vocab_, embd

#mean;   max;    min;
option = args.operation
print('operation:' + option)
trainset = get_doc_vec_options(df_train, word_vector_map, word_embeddings_dim, option)
valset = get_doc_vec_options(df_validation, word_vector_map, word_embeddings_dim, option)
testset = get_doc_vec_options(df_test, word_vector_map, word_embeddings_dim, option)

X_train = np.array(trainset['doc vec'].tolist()); y_train = np.array(train_label)
X_validation = np.array(valset['doc vec'].tolist()); y_validation = np.array(validation_label)
X_test = np.array(testset['doc vec'].tolist()); y_test = np.array(test_label)

#kernel: rbf
best_f1 = 0
kernel = 'rbf'
gammas = [1, 1e-1, 1e-2, 1e-3]
Cs = [1, 10, 100]

for gamma in gammas:
    for c in Cs:
        random_state = 0
        clf = svm.SVC(kernel=kernel, gamma=gamma, C=c, class_weight='balanced', probability=True, random_state=random_state)
        clf.fit(X_train, y_train)
        pred_val = clf.predict(X_validation)
        f1 = f1_score(y_validation, pred_val)
        print('ker:' + str(kernel) + ' gamma:' + str(gamma) + ' c:' + str(c), flush=True)
        print('f1:' + str(f1), flush=True)
        if f1 > best_f1:
            best_f1 = f1; optimal_ker = 'rbf'; optimal_gam = gamma; optimal_c = c; best_clf = clf


kernel = 'linear'
Cs = [1, 10, 100]
for c in Cs:
    random_state = 0
    clf = svm.SVC(kernel=kernel, C=c, class_weight='balanced', probability=True, random_state=random_state)
    clf.fit(X_train, y_train)
    pred_val = clf.predict(X_validation)
    print('ker:' + str(kernel) + ' c:' + str(c), flush = True)
    f1 = f1_score(y_validation, pred_val)
    print('f1:' + str(f1), flush = True)
    if f1 > best_f1:
        best_f1 = f1; optimal_ker = 'linear'; optimal_c = c; best_clf = clf
    print('*'*80)


print('best_f1:' + str(best_f1) + '    optimal_ker:' + str(optimal_ker) + '    optimal_c:' + str(optimal_c) + '    optimal_gamma:' + str(optimal_gam))
print('------------Test Set-----------')
#Save trained model:
output_svm_model_path = './DocSVM/model/' + str(option) + '_finding_model.pkl'
with open(output_svm_model_path,'wb') as f:
    pickle.dump(best_clf,f)

# print('------------probability prediction-----------')
prob_predict = best_clf.predict_proba(X_test)  # <class 'numpy.ndarray'>  shape:(999,2)
np.save(option + '_finding.npy', prob_predict)
prob_label = prob_predict[:, 1].copy()
y_pred = prob_predict[:, 1].copy()
y_pred[y_pred < 0.5] = 0
y_pred[y_pred >= 0.5] = 1
y_true = y_test

print(classification_report(y_true, y_pred, digits=10))
print()
accuracy = accuracy_score(y_true, y_pred)
print("accuracy:" + str(accuracy))
prfs = precision_recall_fscore_support(y_true, y_pred, average='binary')
print("precision_recall_fscore_support:" + str(prfs))
print(confusion_matrix(y_true, y_pred))

import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculating Precision
precision = precision_score(y_true, y_pred)

# Calculating Recall
recall = recall_score(y_true, y_pred)

# Calculating Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculating F1 Score
f1 = f1_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)


