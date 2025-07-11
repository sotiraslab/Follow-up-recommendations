###########################
#   Regex on Impression   #
###########################

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import nltk
nltk.download('punkt_tab')

#keywords
keywords_set1 = ["follow-up", "follow up", "further evaluation", "non-emergent", "non emergent"]
keywords_set2 = ["clinically indicated", "clinically concerned", "continued attention",
                 "attention on", "attention to", "could be considered", "can be considered",
                 "further evaluation can be", "if clinically indicated",
                 "may be helpful", "will be", "follow up exam", "follow up image",
                 "can be further", "no follow up", "no further follow up", "consider",
                 "as clinically warranted", "will follow", "scheduled for" 
                 "planned", "could be further", "could be obtained", "could be performed" ]

# Root directory for the testset
root_directory = ''

#for Test set
test_fu_annotated = root_directory + 'test_fu_annotated.xlsx'
test_nfu_annotated = root_directory + 'test_nfu_annotated.xlsx'
test_templated = root_directory + 'test_templated.xlsx'
test_fu_annotated = pd.read_excel(test_fu_annotated)
test_templated = pd.read_excel(test_templated)
test_nfu_annotated = pd.read_excel(test_nfu_annotated)


gts = []
im_kw_predicted = []
i = 0
for index, row in test_fu_annotated.iterrows():
    
    report = row['Text (Before)']
    #extract impression
    if (report.find('IMPRESSION:') >= 0):
        im = report.split("IMPRESSION:")[-1]
    elif (report.find('Impression:') >= 0):
        im = report.split("Impression:")[-1]
    elif (report.find('IMPRESSION :') >= 0):
        im = report.split("IMPRESSION :")[-1]
    else:
        raise RuntimeError('There is a sample without IMPRESSION part')
        continue
        
    found = any(substring in im for substring in keywords_set1)
    not_found = all(substring not in im for substring in keywords_set2)
    if found and not_found:
        im_kw_predicted.append(1)
    else:
        im_kw_predicted.append(0)
    gts.append(1)


for index, row in test_templated.iterrows():
    
    report = row['Report Text (Before)']
    #extract impression
    if (report.find('IMPRESSION:') >= 0):
        im = report.split("IMPRESSION:")[-1]
    elif (report.find('Impression:') >= 0):
        im = report.split("Impression:")[-1]
    elif (report.find('IMPRESSION :') >= 0):
        im = report.split("IMPRESSION :")[-1]
    else:
        raise RuntimeError('There is a sample without IMPRESSION part')
        
    found = any(substring in im for substring in keywords_set1)
    not_found = all(substring not in im for substring in keywords_set2)
    if found and not_found:
        im_kw_predicted.append(1)
    else:
        im_kw_predicted.append(0)
    gts.append(1)


for index, row in test_nfu_annotated.iterrows():
    
    report = row['Text (Before)']
    #extract impression
    if (report.find('IMPRESSION:') >= 0):
        im = report.split("IMPRESSION:")[-1]
    elif (report.find('Impression:') >= 0):
        im = report.split("Impression:")[-1]
    elif (report.find('IMPRESSION :') >= 0):
        im = report.split("IMPRESSION :")[-1]
    else:
        raise RuntimeError('There is a sample without IMPRESSION part')
        continue
    
    found = any(substring in im for substring in keywords_set1)
    not_found = all(substring not in im for substring in keywords_set2)
    if found and not_found:
        im_kw_predicted.append(1)
    else:
        im_kw_predicted.append(0)
    gts.append(0)

gts = np.array(gts)
im_kw_predicted = np.array(im_kw_predicted)

# Calculating Precision
precision = precision_score(gts, im_kw_predicted)
# Calculating Recall
recall = recall_score(gts, im_kw_predicted)
# Calculating Accuracy
accuracy = accuracy_score(gts, im_kw_predicted)
# Calculating F1 Score
f1 = f1_score(gts, im_kw_predicted)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)