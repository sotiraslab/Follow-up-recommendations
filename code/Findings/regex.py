###########################
#    Regex on Findings    #
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
                 "further evaluation can be", "if clinically indicated", "can be considered" 
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
finding_kw_predicted = []
i = 0
for index, row in test_fu_annotated.iterrows():
    report = row['Report Text']
    #Extract Finding
    if (report.find('IMPRESSION:') >= 0):
        str_wo_impression = report.split("IMPRESSION:")[0]
    elif (report.find('Impression:') >= 0):
        str_wo_impression = report.split("Impression:")[0]
    elif (report.find('IMPRESSION :') >= 0):
        str_wo_impression = report.split("IMPRESSION :")[0]
    else:
        print(str_wo_impression)
        raise RuntimeError('There is a sample without IMPRESSION section')
    if (str_wo_impression.find('FINDINGS:') >= 0):
        sents = str_wo_impression.split("FINDINGS:")[-1]
    elif (str_wo_impression.find('Findings:') >= 0):
        sents = str_wo_impression.split("Findings:")[-1]
    elif (str_wo_impression.find('FINDINGS :') >= 0):
        sents = str_wo_impression.split("FINDINGS :")[-1]
    elif (str_wo_impression.find('FINDINGS') >= 0):
        sents = str_wo_impression.split("FINDINGS")[-1]
    else:
        continue
        
    found = any(substring in sents for substring in keywords_set1)
    not_found = all(substring not in sents for substring in keywords_set2)
    if found and not_found:
        finding_kw_predicted.append(1)
    else:
        finding_kw_predicted.append(0)
    gts.append(1)


for index, row in test_templated.iterrows():
    report = row['Report Text']
    #Extract Finding
    if (report.find('IMPRESSION:') >= 0):
        str_wo_impression = report.split("IMPRESSION:")[0]
    elif (report.find('Impression:') >= 0):
        str_wo_impression = report.split("Impression:")[0]
    elif (report.find('IMPRESSION :') >= 0):
        str_wo_impression = report.split("IMPRESSION :")[0]
    else:
        print(str_wo_impression)
        raise RuntimeError('There is a sample without IMPRESSION section')
    if (str_wo_impression.find('FINDINGS:') >= 0):
        sents = str_wo_impression.split("FINDINGS:")[-1]
    elif (str_wo_impression.find('Findings:') >= 0):
        sents = str_wo_impression.split("Findings:")[-1]
    elif (str_wo_impression.find('FINDINGS :') >= 0):
        sents = str_wo_impression.split("FINDINGS :")[-1]
    elif (str_wo_impression.find('FINDINGS') >= 0):
        sents = str_wo_impression.split("FINDINGS")[-1]
    else:
        continue
        
    found = any(substring in sents for substring in keywords_set1)
    not_found = all(substring not in sents for substring in keywords_set2)
    if found and not_found:
        finding_kw_predicted.append(1)
    else:
        finding_kw_predicted.append(0)
    gts.append(1)


for index, row in test_nfu_annotated.iterrows():
    report = row['Report Text']
    #Extract Finding
    if (report.find('IMPRESSION:') >= 0):
        str_wo_impression = report.split("IMPRESSION:")[0]
    elif (report.find('Impression:') >= 0):
        str_wo_impression = report.split("Impression:")[0]
    elif (report.find('IMPRESSION :') >= 0):
        str_wo_impression = report.split("IMPRESSION :")[0]
    else:
        print(str_wo_impression)
        raise RuntimeError('There is a sample without IMPRESSION section')
    if (str_wo_impression.find('FINDINGS:') >= 0):
        sents = str_wo_impression.split("FINDINGS:")[-1]
    elif (str_wo_impression.find('Findings:') >= 0):
        sents = str_wo_impression.split("Findings:")[-1]
    elif (str_wo_impression.find('FINDINGS :') >= 0):
        sents = str_wo_impression.split("FINDINGS :")[-1]
    elif (str_wo_impression.find('FINDINGS') >= 0):
        sents = str_wo_impression.split("FINDINGS")[-1]
    else:
        continue
    
    found = any(substring in sents for substring in keywords_set1)
    not_found = all(substring not in sents for substring in keywords_set2)
    if found and not_found:
        finding_kw_predicted.append(1)
    else:
        finding_kw_predicted.append(0)
    gts.append(0)

gts = np.array(gts)
finding_kw_predicted = np.array(finding_kw_predicted)

# Calculating Precision
precision = precision_score(gts, finding_kw_predicted)
# Calculating Recall
recall = recall_score(gts, finding_kw_predicted)
# Calculating Accuracy
accuracy = accuracy_score(gts, finding_kw_predicted)
# Calculating F1 Score
f1 = f1_score(gts, finding_kw_predicted)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)