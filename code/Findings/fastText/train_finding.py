import fasttext
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np

model = fasttext.train_supervised(input='./Finding/Train/Radreport.train', lr = 0.1, epoch = 10, minCount = 5 , dim=300, autotuneValidationFile='./Finding/Validation/Radreport.validation')
print(model.labels)

test_list = open("./Finding/Test/Radreport.test",'r')

true_labels = []
pred_labels = []
proba_labels = []
for line in test_list:
    true_label = line.split('\t')[0]
    #print(true_label)
    if true_label == '__label__nofollowup':
        true_labels.append(0)
    elif true_label == '__label__followup':
        true_labels.append(1)
    text = ' '.join(line.split('\t')[1:]).strip()
    a, b = model.predict(text, k=2)
    #print(a[0])
    if a[0] == '__label__nofollowup':
        pred_labels.append(0)
        proba_labels.append(1-b[0])
    elif a[0] == '__label__followup':
        pred_labels.append(1)
        proba_labels.append(b[0])

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

model.save_model("ft_finding_300.bin")
test_list.close()
proba_labels = np.array(proba_labels)
np.save('fasttext_finding_300_w_modelsaved.npy', proba_labels)
proba_labels[proba_labels>=0.5] = 1
proba_labels[proba_labels<0.5] = 0
prfs = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
print(prfs)
print(accuracy_score(true_labels, pred_labels))
print(confusion_matrix(true_labels, pred_labels))
print(confusion_matrix(true_labels, proba_labels.tolist()))

import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to the system path
sys.path.insert(0, parent_dir)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Calculating Precision
precision = precision_score(true_labels, pred_labels)

# Calculating Recall
recall = recall_score(true_labels, pred_labels)

# Calculating Accuracy
accuracy = accuracy_score(true_labels, pred_labels)

# Calculating F1 Score
f1 = f1_score(true_labels, pred_labels)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)

    