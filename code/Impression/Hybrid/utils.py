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

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def text_cleaner(text):
    rules = [
        #{r'\[': u''},
        #{r'\]': u''},
        #{r',.;': u''},
        {r'[^a-zA-Z\s]': u''},
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}, # remove spaces at the beginning
        {r'[\xa0]':u''},
    ]
    '''
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    '''
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
    text = text.strip()
    stop_words = set(stopwords.words('english'))
    text = ' '.join(ww for ww in text.split() if not ww in stop_words)
    return text

#One sentence
def converttoBertFormat(filename, fileformat, posorneg):
    # fileformat = 'xls' or 'csv'
    if fileformat == 'xls':
        df = pd.read_excel(filename)
    else:
        df = pd.read_csv(filename, encoding='cp1252')
    Impression = []
    pdImpression = pd.DataFrame(columns=['Report Text'])
    max_num_words = 0
    for i in range(0, df['Report Text'].count()):
        sent_of_each_document = []
        str = df['Report Text'][i]
        # tokens = word_tokenize(str)
        if (str.find('IMPRESSION:') >= 0):
            sents = str.split("IMPRESSION:")[-1]
        elif (str.find('Impression:') >= 0):
            sents = str.split("Impression:")[-1]
        elif (str.find('IMPRESSION :') >= 0):
            sents = str.split("IMPRESSION :")[-1]
        else:
            print(i)
            print(sents)
            raise RuntimeError('There is a sample without IMPRESSION part')
        # sent = sents[1]
        if (sents.find('Dictated by') >= 0):
            sent = sents.split('Dictated by')[0]
        elif (sents.find('Electronically signed by') >= 0):
            sent = sents.split('Electronically signed by')[0]
        elif (sents.find('These images are for Reference purposes only') >= 0):
            sent = sents.split('These images are for Reference purposes only')[0]
        else:
            sent = sents.split('Dictated by')[0]
            # print(i)
            # print(sents)
            # raise RuntimeError('There is a sample without Dictated by')
        sent = sent.replace("\n", " ")
        sent = sent.split(".")
        num_words = 0
        for subsent in sent:
            if subsent == ' ' or '':
                continue
            subsent = subsent.lower()
            subsent = re.sub(r'follow[a-z ]{1,5}up', 'followup', subsent)
            # subsent = subsent.replace("follow up","followup")
            tokens = word_tokenize(subsent)
            tokens = ' '.join(word for word in tokens)
            tokens = text_cleaner(tokens) + '.'
            num_words += len(tokens)
            if tokens == ".":
                continue
            else:
                sent_of_each_document.append(tokens)
        all_sent_of_each_document =' '.join(sent_of_each_document)
        all_sent_of_each_document = all_sent_of_each_document.replace(". ", " [period] ")
        all_sent_of_each_document = all_sent_of_each_document.replace(".", " [period]")
        all_sent_of_each_document = all_sent_of_each_document.replace(", ", " [comma] ")
        if num_words > max_num_words:
          max_num_words = num_words
        pdImpression.loc[i] = all_sent_of_each_document
        
    return pdImpression, max_num_words

def exttokens(df, tce, num_ext, posorneg):
    
    pdImpression = pd.DataFrame(columns=['Report Text'])
    countword = []
    for i in range(0, df['Report Text'].count()):
        str = df['Report Text'][i]
        tokens = word_tokenize(str)
        num_words = len(tokens)
        countword.append(num_words)
        if num_words > num_ext:
            if tce == 'top':
                tokens = tokens[0:num_ext]
            elif tce == 'center':
                tokens = tokens[int(num_words/2)-int(num_ext/2): int(num_words/2) + int(num_ext/2)]
            elif tce == 'end':
                tokens = tokens[num_words-num_ext:num_words]
        if posorneg == 'pos':
            tokens ='1\t' + ' '.join(word for word in tokens)
        elif posorneg == 'neg':
            tokens ='0\t' + ' '.join(word for word in tokens)
        pdImpression.loc[i] = tokens
    return pdImpression, countword

def convertpdtofile(df, filepath):
    str = ''
    with open(filepath, 'wt') as out_file:
        for i in range(0, df['Report Text'].count()):
            str = df['Report Text'][i] + '\n'
            out_file.write(str)
    out_file.close()


def convertfiletoImpression(filename, fileformat):
    #fileformat = 'xls' or 'csv'
    if fileformat == 'xls':
        df = pd.read_excel(filename)
    else:
        df = pd.read_csv(filename, encoding = 'cp1252')
    Impression = []
    print("The number of Report Text:", df['Report Text'].count())
    for i in range(0, df['Report Text'].count()):
        sent_of_each_document = []
        str = df['Report Text'][i]
        # tokens = word_tokenize(str)
        if (str.find('IMPRESSION:') >= 0):
            sents = str.split("IMPRESSION:")[-1]
        elif (str.find('Impression:') >= 0):
            sents = str.split("Impression:")[-1]
        elif (str.find('IMPRESSION :') >= 0):
            sents = str.split("IMPRESSION :")[-1]
        else:
            print(i)
            print(sents)
            raise RuntimeError('There is a sample without IMPRESSION part')
        # sent = sents[1]
        if (sents.find('Dictated by') >= 0):
            sent = sents.split('Dictated by')[0]
        elif (sents.find('Electronically signed by') >= 0):
            sent = sents.split('Electronically signed by')[0]
        elif (sents.find('These images are for Reference purposes only') >= 0):
            sent = sents.split('These images are for Reference purposes only')[0]
        else:
            sent = sents.split('Dictated by')[0]
            #print(i)
            #print(sents)
            # raise RuntimeError('There is a sample without Dictated by')
        sent = sent.replace("\n", " ")
        #sent = sent.replace(". ", " [period] ")
        #sent = sent.replace(", ", " [comma] ")
        sent = sent.split(".")
        for subsent in sent:
            subsent = subsent.strip()
            if subsent == ' ' or '' or len(subsent) < 2:
                continue
            subsent = subsent.lower()
            subsent = re.sub(r'follow[a-z ]{1,5}up', 'followup', subsent)
            # subsent = subsent.replace("follow up","followup")
            tokens = word_tokenize(subsent)
            tokens = ' '.join(word for word in tokens)
            tokens = text_cleaner(tokens) + '.'
            if tokens == ".":
                continue
            else:
                sent_of_each_document.append(tokens)
        all_sent_of_each_document = ' '.join(sent_of_each_document) #split every sentence.
        all_sent_of_each_document = all_sent_of_each_document.replace(". ", " [period] ")
        all_sent_of_each_document = all_sent_of_each_document.replace(", ", " [comma] ")
        Impression.append(all_sent_of_each_document)
    Impression_all = '\n'.join(Impression)
    Impression_all = Impression_all.replace(".\n", " [period]\n")
    # Output the impression part.
    # Impressions are split by '\n'. Sentences in Impression part are split by ' '.
    return Impression_all

def convertfiletoInput(filename, fileformat, format):
    #fileformat = 'xls' or 'csv'
    if fileformat == 'xls':
        df = pd.read_excel(filename)
    else:
        df = pd.read_csv(filename, encoding = 'cp1252')
    pdImpression = pd.DataFrame(columns=['Report Text'])

    print("The number of Report Text:", df['Report Text'].count())
    for i in range(0, df['Report Text'].count()):
        sent_of_each_document = []
        str = df['Report Text'][i]
        # tokens = word_tokenize(str)
        if (str.find('IMPRESSION:') >= 0):
            sents = str.split("IMPRESSION:")[-1]
        elif (str.find('Impression:') >= 0):
            sents = str.split("Impression:")[-1]
        elif (str.find('IMPRESSION :') >= 0):
            sents = str.split("IMPRESSION :")[-1]
        else:
            print(i)
            print(sents)
            raise RuntimeError('There is a sample without IMPRESSION part')
        # sent = sents[1]
        if (sents.find('Dictated by') >= 0):
            sent = sents.split('Dictated by')[0]
        elif (sents.find('Electronically signed by') >= 0):
            sent = sents.split('Electronically signed by')[0]
        elif (sents.find('These images are for Reference purposes only') >= 0):
            sent = sents.split('These images are for Reference purposes only')[0]
        else:
            sent = sents.split('Dictated by')[0]
            #print(i)
            #print(sents)
            # raise RuntimeError('There is a sample without Dictated by')
        sent = sent.replace("\n", " ")
        #sent = sent.replace(". ", " [period] ")
        #sent = sent.replace(", ", " [comma] ")
        sent = sent.split(".")
        for subsent in sent:
            subsent = subsent.strip()
            if subsent == ' ' or '' or len(subsent) < 2:
                continue
            subsent = subsent.lower()
            subsent = re.sub(r'follow[a-z ]{1,5}up', 'followup', subsent)
            # subsent = subsent.replace("follow up","followup")
            tokens = word_tokenize(subsent)
            tokens = ' '.join(word for word in tokens)
            tokens = text_cleaner(tokens) + ' . '
            if tokens == ".":
                continue
            else:
                sent_of_each_document.append(tokens)
        all_sent_of_each_document = ''.join(sent_of_each_document) #split every sentence.
        all_sent_of_each_document = all_sent_of_each_document.replace(". ", "[period] ")
        if format == 'pos':
            all_sent_of_each_document = '2\t' +all_sent_of_each_document
        elif format == 'soft':
            all_sent_of_each_document = '1\t' +all_sent_of_each_document
        elif format == 'neg':
            all_sent_of_each_document = '0\t' +all_sent_of_each_document
        pdImpression.loc[i] = [all_sent_of_each_document.strip()]
        #if i > 100:
        #    break
    # Output the impression part.
    # Impressions are split by '\n'. Sentences in Impression part are split by ' '.
    return pdImpression

def readtestimpretoDataframe(filename):
    dftest = pd.read_csv(filename, encoding = 'cp1252')
    pdImpression = pd.DataFrame(columns=['Accession Number', 'Report Text'])

    for i in range(0, dftest['Report Text'].count()):
        sent_of_each_document = []
        str_rep = dftest['Report Text'][i]
        ascession = dftest['Accession Number'][i]
        # tokens = word_tokenize(str)
        if (str_rep.find('IMPRESSION:') >= 0):
            sents = str_rep.split("IMPRESSION:")[-1]
        elif (str_rep.find('Impression:') >= 0):
            sents = str_rep.split("Impression:")[-1]
        elif (str_rep.find('IMPRESSION :') >= 0):
            sents = str_rep.split("IMPRESSION :")[-1]
        else:
            print(i)
            print(sents)
            raise RuntimeError('There is a sample without IMPRESSION part')
        # sent = sents[1]
        if (sents.find('These images are for Reference purposes only') >= 0):
            sent = sents.split('These images are for Reference purposes only')[0]
        elif (sents.find('Dictated by') >= 0):
            sent = sents.split('Dictated by')[0]
        elif (sents.find('Electronically signed by') >= 0):
            sent = sents.split('Electronically signed by')[0]
        else:
            sent = sents
            
        sent = sent.replace("\n", " ")
        sent = sent.split(".")
        for subsent in sent:
            if subsent == ' ' or '':
                continue
            subsent = subsent.lower()
            subsent = re.sub(r'follow[a-z ]{1,5}up', 'followup', subsent)
            # subsent = subsent.replace("follow up","followup")
            tokens = word_tokenize(subsent)
            tokens = ' '.join(word for word in tokens)
            tokens = text_cleaner(tokens) + '.'
            if tokens == ".":
                continue
            else:
                sent_of_each_document.append(tokens)
        if len(sent_of_each_document) == 0:   ###if you want to determine if a list is in vain, the command "sent_of_each_document == '' " doesn't work, you should use "len(sent_of_each_document) == 0"
            print('find a text report without impression part\n')
            continue
        all_sent_of_each_document = ' '.join(sent_of_each_document)
        pdImpression.loc[i] = [dftest['Accession Number'][i], all_sent_of_each_document]
        
    pdImpression = pdImpression.reset_index(drop=True)
    return pdImpression, dftest


def readimpressiontoDataframe(filename, fileformat):
    # fileformat = 'xls' or 'csv'
    if fileformat == 'xls':
        dffile = pd.read_excel(filename)
    else:
        dffile = pd.read_csv(filename, encoding='cp1252')

    #dffile = pd.read_excel(filename)
    dffile = dffile.reset_index(drop=True)
    pdImpression = pd.DataFrame(columns=['Accession Number', 'Report Text'])

    for i in range(0, dffile['Report Text'].count()):
        sent_of_each_document = []
        str_rep = dffile['Report Text'][i]
        ascession = dffile['Accession Number'][i]
        # tokens = word_tokenize(str)
        if (str_rep.find('IMPRESSION:') >= 0):
            sents = str_rep.split("IMPRESSION:")[-1]
        elif (str_rep.find('Impression:') >= 0):
            sents = str_rep.split("Impression:")[-1]
        elif (str_rep.find('IMPRESSION :') >= 0):
            sents = str_rep.split("IMPRESSION :")[-1]
        else:
            print(i)
            print(sents)
            raise RuntimeError('There is a sample without IMPRESSION part')
        # get Impression part (sent)
        sent = sents
        sent = sent.replace("\n", " ")
        sent = sent.split(".")
        for subsent in sent:
            if subsent == ' ' or '':
                continue
            subsent = subsent.lower()
            subsent = re.sub(r'follow[a-z ]{1,5}up', 'followup', subsent)
            tokens = word_tokenize(subsent)
            tokens = ' '.join(word for word in tokens)
            tokens = text_cleaner(tokens) + '.'
            if tokens == ".":
                continue
            else:
                sent_of_each_document.append(tokens)
        # all_sent_of_each_document is an Impression
        all_sent_of_each_document = ' '.join(sent_of_each_document)
        pdImpression.loc[i] = [dffile['Accession Number'][i], all_sent_of_each_document]
        #if i > 100:
        #    break

    return pdImpression


def split_train_val_test(dfpossave, dfnegsave, ratio):
    #ratio represents how much the training set will occupy
    poslines = dfpossave['Report Text']
    neglines = dfnegsave['Report Text']

    neglines.index = range(0, len(neglines))

    rdlens = [i for i in range(0, len(neglines))]
    random.shuffle(rdlens)
    dfnegsave = dfnegsave.reindex(np.array(rdlens))
    negsave = dfnegsave.reset_index(drop=True)

    rdlens = [i for i in range(0, len(poslines))]
    random.shuffle(rdlens)
    dfpossave = dfpossave.reindex(np.array(rdlens))
    possave = dfpossave.reset_index(drop=True)

    neglens = len(neglines)
    poslens = len(poslines)
    print("neglens:"+str(neglens))
    print("poslens:"+str(poslens))
    weights = poslens/(neglens+1)
    #0.5 : 0.2 : 0.3
    #raise RuntimeError("break")

    negtrain_Ind = int(neglens*ratio)
    #negval_Ind = int(neglens*0.2)
    postrain_Ind = int(poslens*ratio)
    #posval_Ind = int(poslens*0.2)

    negtrain = negsave[0: negtrain_Ind]
    #negval = neglines[negtrain_Ind: negtrain_Ind+negval_Ind]
    negtest = negsave[negtrain_Ind:]

    postrain = possave[0: postrain_Ind]
    #posval = poslines[postrain_Ind: postrain_Ind + posval_Ind]
    postest = possave[postrain_Ind:]

    trainset = pd.concat([postrain,negtrain])
    train_label = [1 if i < len(postrain) else 0 for i in range(0,len(trainset))]
    wieghts_label = [1 if i < len(postrain) else weights for i in range(0, len(trainset))]
    trainset['label'] = train_label
    trainset['weights_label'] = wieghts_label
    trainset = trainset.reset_index(drop=True)

    testset = pd.concat([postest, negtest])
    test_label = [1 if i < len(postest) else 0 for i in range(0, len(testset))]
    testset['label'] = test_label
    testset = testset.reset_index(drop=True)

    rdlens = [i for i in range(0, len(trainset))]
    random.shuffle(rdlens)
    trainset = trainset.reindex(np.array(rdlens))
    trainset = trainset.reset_index(drop=True)

    rdlens = [i for i in range(0, len(testset))]
    random.shuffle(rdlens)
    testset = testset.reindex(np.array(rdlens))
    testset = testset.reset_index(drop=True)

    return trainset, testset

def split_train_test_based_Ind(dfpos, dfneg, pos_train_ind_file, pos_test_ind_file, neg_train_ind_file, neg_test_ind_file):
    with open(pos_train_ind_file, 'rb') as filehandle:
        pos_train_ind = pickle.load(filehandle)
    with open(pos_test_ind_file, 'rb') as filehandle:
        pos_test_ind = pickle.load(filehandle)
    with open(neg_train_ind_file, 'rb') as filehandle:
        neg_train_ind = pickle.load(filehandle)
    with open(neg_test_ind_file, 'rb') as filehandle:
        neg_test_ind = pickle.load(filehandle)

    train_pos = dfpos.iloc[pos_train_ind, :]
    test_pos = dfpos.iloc[pos_test_ind, :]
    train_neg = dfneg.iloc[neg_train_ind, :]
    test_neg = dfneg.iloc[neg_test_ind, :]

    trainset = pd.concat([train_pos, train_neg])
    train_label = [1 if i < len(train_pos) else 0 for i in range(0, len(trainset))]
    trainset['label'] = train_label
    trainset = trainset.reset_index(drop=True)

    testset = pd.concat([test_pos, test_neg])
    test_label = [1 if i < len(test_pos) else 0 for i in range(0, len(testset))]
    testset['label'] = test_label
    testset = testset.reset_index(drop=True)

    rdlens = [i for i in range(0, len(trainset))]
    random.shuffle(rdlens, random.seed(1))
    trainset = trainset.reindex(np.array(rdlens))
    trainset = trainset.reset_index(drop=True)

    rdlens = [i for i in range(0, len(testset))]
    random.shuffle(rdlens, random.seed(1))
    testset = testset.reindex(np.array(rdlens))
    testset = testset.reset_index(drop=True)

    return trainset, testset

def split_and_save_train_dev_test_based_Ind(filepos, filesoft, fileneg, pos_train_ind_file, pos_dev_ind_file, pos_test_ind_file, soft_train_ind_file, soft_dev_ind_file, soft_test_ind_file, neg_train_ind_file, neg_dev_ind_file, neg_test_ind_file):

    dfpos = convertfiletoInput(filepos, 'xls', 'pos')
    dfneg = convertfiletoInput(fileneg, 'xls', 'neg')
    dfsoft = convertfiletoInput(filesoft, 'xls', 'soft')

    # load
    with open(pos_train_ind_file, 'rb') as filehandle:
        pos_train_ind = pickle.load(filehandle)
    with open(pos_dev_ind_file, 'rb') as filehandle:
        pos_dev_ind = pickle.load(filehandle)
    with open(pos_test_ind_file, 'rb') as filehandle:
        pos_test_ind = pickle.load(filehandle)
    with open(soft_train_ind_file, 'rb') as filehandle:
        soft_train_ind = pickle.load(filehandle)
    with open(soft_dev_ind_file, 'rb') as filehandle:
        soft_dev_ind = pickle.load(filehandle)
    with open(soft_test_ind_file, 'rb') as filehandle:
        soft_test_ind = pickle.load(filehandle)
    with open(neg_train_ind_file, 'rb') as filehandle:
        neg_train_ind = pickle.load(filehandle)
    with open(neg_dev_ind_file, 'rb') as filehandle:
        neg_dev_ind = pickle.load(filehandle)
    with open(neg_test_ind_file, 'rb') as filehandle:
        neg_test_ind = pickle.load(filehandle)

    train_pos = dfpos.iloc[pos_train_ind, :]
    dev_pos = dfpos.iloc[pos_dev_ind, :]
    test_pos = dfpos.iloc[pos_test_ind, :]

    train_soft = dfsoft.iloc[soft_train_ind, :]
    dev_soft = dfsoft.iloc[soft_dev_ind, :]
    test_soft = dfsoft.iloc[soft_test_ind, :]

    train_neg = dfneg.iloc[neg_train_ind, :]
    dev_neg = dfneg.iloc[neg_dev_ind, :]
    test_neg = dfneg.iloc[neg_test_ind, :]

    trainset = pd.concat([train_pos, train_soft, train_neg])
    trainset = trainset.reset_index(drop=True)

    devset = pd.concat([dev_pos, dev_soft, dev_neg])
    devset = devset.reset_index(drop=True)

    testset = pd.concat([test_pos, test_soft, test_neg])
    testset = testset.reset_index(drop=True)

    rdlens = [i for i in range(0, len(trainset))]
    random.shuffle(rdlens, random.seed(1))
    trainset = trainset.reindex(np.array(rdlens))
    trainset = trainset.reset_index(drop=True)

    rdlens = [i for i in range(0, len(devset))]
    random.shuffle(rdlens, random.seed(1))
    devset = devset.reindex(np.array(rdlens))
    devset = devset.reset_index(drop=True)

    rdlens = [i for i in range(0, len(testset))]
    random.shuffle(rdlens, random.seed(1))
    testset = testset.reindex(np.array(rdlens))
    testset = testset.reset_index(drop=True)

    trainset = trainset['Report Text'].str.cat(sep='\n')
    devset = devset['Report Text'].str.cat(sep='\n')
    testset = testset['Report Text'].str.cat(sep='\n')

    return trainset, devset, testset

#
def split_train_eval_test(dfpossave, dfnegsave, ratiotrain, ratioeval):
    #ratio represents how much the training set will occupy
    poslines = dfpossave['Report Text']
    neglines = dfnegsave['Report Text']

    neglines.index = range(0, len(neglines))

    rdlens = [i for i in range(0, len(neglines))]
    random.shuffle(rdlens)
    dfnegsave = dfnegsave.reindex(np.array(rdlens))
    negsave = dfnegsave.reset_index(drop=True)

    rdlens = [i for i in range(0, len(poslines))]
    random.shuffle(rdlens)
    dfpossave = dfpossave.reindex(np.array(rdlens))
    possave = dfpossave.reset_index(drop=True)

    neglens = len(neglines)
    poslens = len(poslines)
    print("neglens:"+str(neglens))
    print("poslens:"+str(poslens))
    #weights = poslens/(neglens+1)
    #0.5 : 0.2 : 0.3
    #raise RuntimeError("break")

    negtrain_Ind = int(neglens*ratiotrain)
    negeval_Ind = int(neglens*ratioeval)
    postrain_Ind = int(poslens*ratiotrain)
    poseval_Ind = int(poslens*ratioeval)

    negtrain = negsave[0: negtrain_Ind]
    negeval = negsave[negtrain_Ind: negtrain_Ind+negeval_Ind]
    negtest = negsave[negtrain_Ind+negeval_Ind:]

    postrain = possave[0: postrain_Ind]
    poseval = possave[postrain_Ind: postrain_Ind + poseval_Ind]
    postest = possave[postrain_Ind + poseval_Ind:]

    trainset = pd.concat([postrain,negtrain])
    #train_label = [1 if i < len(postrain) else 0 for i in range(0,len(trainset))]
    #wieghts_label = [1 if i < len(postrain) else weights for i in range(0, len(trainset))]
    #trainset['label'] = train_label
    #trainset['weights_label'] = wieghts_label
    trainset = trainset.reset_index(drop=True)

    evalset = pd.concat([poseval, negeval])
    evalset = evalset.reset_index(drop=True)

    testset = pd.concat([postest, negtest])
    #test_label = [1 if i < len(postest) else 0 for i in range(0, len(testset))]
    #testset['label'] = test_label
    testset = testset.reset_index(drop=True)

    rdlens = [i for i in range(0, len(trainset))]
    random.shuffle(rdlens)
    trainset = trainset.reindex(np.array(rdlens))
    trainset = trainset.reset_index(drop=True)

    rdlens = [i for i in range(0, len(evalset))]
    random.shuffle(rdlens)
    evalset = evalset.reindex(np.array(rdlens))
    evalset = evalset.reset_index(drop=True)

    rdlens = [i for i in range(0, len(testset))]
    random.shuffle(rdlens)
    testset = testset.reindex(np.array(rdlens))
    testset = testset.reset_index(drop=True)

    return trainset, evalset, testset

#save embed、 vocab to model_googlew2v
def savemodelfile(WV, fo, binary):
    syn0 = WV._syn0_final
    vocab = WV._vocab

    print('Saving model_googlew2v to', fo)
    dim = len(syn0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('%d %d\n' % (len(syn0), dim))
        fo.write('\n')
        for token, vector in zip(vocab, syn0):
            fo.write('%s ' % token.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(vocab, syn0):
            word = token
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (word, vector_str))

    fo.close()


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


def get_doc_vec(dfset, word_vector_map, word_embeddings_dim):

    real_size = len(dfset)
    data_all_word_vec = []
    for i in range(0, real_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        doc_words = dfset['Report Text'][i]  # one document
        if type(doc_words) is float:
            print(doc_words)
            doc_words = str(doc_words)
        doc_words = doc_words.replace('.', ' ')
        doc_words = doc_words.replace(',', ' ')
        words = word_tokenize(doc_words)  # all words in one document
        doc_len = len(words)  # the number of words in one document
        wordcount = 0
        for word in words:
            if word in word_vector_map:
                wordcount = wordcount + 1
                word_vector = word_vector_map[word]
                # print(doc_vec)
                # print(np.array(word_vector))
                doc_vec = doc_vec + np.array(word_vector)  # add all word vector to represent doc vec
        if wordcount != 0:
            doc_vec = doc_vec / wordcount
            data_all_word_vec.append(doc_vec)
        else: # delete the null samples
            print(i)
            print('\n')
            dfset = dfset.drop(i)
    dfset = dfset.reset_index(drop=True)    #reset the index (https://pandas.pydata.org/pandas-docs/version/0.22.0/generated/pandas.DataFrame.reset_index.html)
    dfset['doc vec'] = data_all_word_vec

    return  dfset

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
            raise Exception('Impression part is null')
            print("Impression part is null, delete the null samples")
            print(i)
            print('\n')
            dfset = dfset.drop(i)
    dfset = dfset.reset_index(drop=True)    #reset the index (https://pandas.pydata.org/pandas-docs/version/0.22.0/generated/pandas.DataFrame.reset_index.html)
    dfset['doc vec'] = data_all_word_vec

    return  dfset


# get vocabulary-idf dict
def vocidf(vocabulary, idf_):
    voc_idf = {}
    for key, value in vocabulary.items():
        voc_idf[key] = idf_[value]

    return voc_idf

#option = [mean, max, maxmin]
#the length of doc is smaller than 15, no slicing;;;;
def get_doc_vec_options_idf(dfset, word_vector_map, word_embeddings_dim, option, vocidf, ratio):
    real_size = len(dfset)
    data_all_word_vec = []
    for i in range(0, real_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        # initial doc vector to get min
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
        wordsidf = {}
        for word in words:
            if word in vocidf.keys():
                wordsidf[word] = vocidf[word]
        if len(wordsidf) > 10:
            wordsidf = collections.Counter(wordsidf).most_common()
            wordsidf_per = math.ceil(len(wordsidf) * ratio)
            wordsidf_slice = wordsidf[:wordsidf_per]
            doc_len = len(wordsidf_slice)  # the number of words in one document
        else:
            wordsidf = collections.Counter(wordsidf).most_common()
            wordsidf_per = math.ceil(len(wordsidf) * 1)
            wordsidf_slice = wordsidf[:wordsidf_per]
            doc_len = len(wordsidf_slice)
        if doc_len == 0:
            print("there is a sample without words after slicing:", i)
            print(wordsidf_slice)
        wordcount = 0
        for word, idf in wordsidf_slice:
            if word in word_vector_map:
                wordcount = wordcount + 1
                # contain numerical positive and negative values
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
            #else:
                #print("Some words don't exist in our model_googlew2v")
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
        else:  # delete the null samples
            raise Exception('Impression part after slicing is null')
            print("Impression part is null, delete the null samples")
            print(i)
            print('\n')
            dfset = dfset.drop(i)
    dfset = dfset.reset_index(
        drop=True)  # reset the index (https://pandas.pydata.org/pandas-docs/version/0.22.0/generated/pandas.DataFrame.reset_index.html)
    dfset['doc vec'] = data_all_word_vec

    return dfset

#option = [mean]
def get_doc_vec_options_idf_weighted(dfset, word_vector_map, word_embeddings_dim, option, vocidf, ratio):

    real_size = len(dfset)
    data_all_word_vec = []
    for i in range(0, real_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        doc_words = dfset['Report Text'][i]  # one document
        if type(doc_words) is float:
            print(doc_words)
            doc_words = str(doc_words)
        doc_words = doc_words.replace('.', ' ')
        doc_words = doc_words.replace(',', ' ')
        words = word_tokenize(doc_words) # all words in one document
        wordsidf = {}
        for word in words:
            if word in vocidf.keys():
                wordsidf[word] = vocidf[word]
        wordsidf = collections.Counter(wordsidf).most_common()
        wordsidf_per = math.ceil(len(wordsidf) * ratio)
        wordsidf_slice = wordsidf[:wordsidf_per]
        doc_len = len(wordsidf_slice)  # the number of words in one document
        weight = 0
        for word, idf in wordsidf_slice:
            if word in word_vector_map:
                weight = weight + idf
                #contain numerical positive and negative values
                word_vector = word_vector_map[word]
                # print(doc_vec)
                # print(np.array(word_vector))
                if option == 'mean':
                    doc_vec = doc_vec + np.array(word_vector)*idf  # add all word vector to represent doc vec
        if weight != 0:
            if option == 'mean':
                doc_vec = doc_vec / weight
                data_all_word_vec.append(doc_vec)
        else: # delete the null samples
            raise Exception('Impression part is null')
            print("Impression part is null, delete the null samples")
            print(i)
            print('\n')
            dfset = dfset.drop(i)
    dfset = dfset.reset_index(drop=True)    #reset the index (https://pandas.pydata.org/pandas-docs/version/0.22.0/generated/pandas.DataFrame.reset_index.html)
    dfset['doc vec'] = data_all_word_vec

    return  dfset

def cf_precision_Recall_Accuracy_F1(y_test, y_pred):

    # Calculate the confusion matrix
    #
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('Confusion Matrix:' % precision_score(y_test, y_pred))
    #
    # Print the confusion matrix using Matplotlib
    #
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    #
    # plt.xlabel('Predictions', fontsize=18)
    # plt.ylabel('Actuals', fontsize=18)
    # plt.title('Confusion Matrix', fontsize=18)
    # plt.show()
    #The same score can be obtained by using precision_score method from sklearn.metrics
    print('Precision: %.3f' % precision_score(y_test, y_pred))
    #The same score can be obtained by using recall_score method from sklearn.metrics
    print('Recall: %.3f' % recall_score(y_test, y_pred))
    #The same score can be obtained by using accuracy_score method from sklearn.metrics
    print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    #The same score can be obtained by using f1_score method from sklearn.metrics
    print('F1 Score: %.3f' % f1_score(y_test, y_pred))