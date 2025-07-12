import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import KFold
import math
import struct
from nltk.corpus import stopwords
import random

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
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
    string = re.sub(r"\.", " . ", string)

    stop_words = set(stopwords.words('english'))
    string = ' '.join(ww for ww in string.split() if not ww in stop_words)
    return string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_datasets_Radreport(seed):
    #filenames
    trainind = './data/Radreport_Impression/Radreport_Impression.train.index'
    testind = './data/Radreport_Impression/Radreport_Impression.test.index'
    split_labels = './data/Radreport_Impression/Radreport_Impression.txt'
    cleaneddata = './data/Radreport_Impression/Radreport_Impression.clean.txt'
    #read files
    cleaned_data = []
    f = open(cleaneddata, 'rb')
    for line in f.readlines():
        cleaned_data.append(line.strip().decode('latin1'))
    f.close()
    #read train index
    train_ind = []
    f = open(trainind, 'rb')
    for line in f.readlines():
        train_ind.append(line.strip().decode('latin1'))
    f.close()
    #read test index
    test_ind = []
    f = open(testind, 'rb')
    for line in f.readlines():
        test_ind.append(line.strip().decode('latin1'))
    f.close()
    #read labels
    labels = []
    f = open(split_labels, 'r')
    for line in f.readlines():
        labels.append(line.split("\t")[2].strip())
    f.close()

    train = []; test = []; train_label = []; test_label = []
    for i in range(len(labels)):
        if str(i) in train_ind:
            train.append(cleaned_data[i]); train_label.append(labels[i])
        elif str(i) in test_ind:
            test.append(cleaned_data[i]); test_label.append(labels[i])
    import random
    random.seed(seed)
    c = list(zip(train, train_label)); random.shuffle(c); train, train_label = zip(*c)
    c = list(zip(test, test_label)); random.shuffle(c); test, test_label = zip(*c)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_target = le.fit_transform(train_label)
    test_target = le.transform(test_label)
    from sklearn.utils import Bunch

    Tr = Bunch(data = train, target = train_target, target_names=le.classes_)
    Te = Bunch(data = test, target = test_target, target_names=le.classes_)

    return Tr, Te

def get_datasets_Radreport_impression_tvt(fold):
    #filenames
    #trainind = './data/Radreport_Impression/Radreport.train.index'
    #testind = './data/Radreport_Impression/Radreport.test.index'
    #split_labels = './data/Radreport_Impression/Radreport.txt'
    #cleaneddata = './data/Radreport_Impression/Radreport.clean.txt'
    train = './data/Radreport_Impression/train.txt'
    validation = './data/Radreport_Impression/validation_' + str(fold) + '.txt'
    test = './data/Radreport_Impression/test_' + str(fold) + '.txt'

    #read validation
    val_data = []; val_label = []
    f_val = open(validation, 'rb')
    for line in f_val.readlines():
        val_data.append(line.strip().decode('latin1').split('\t')[1]); val_label.append(line.strip().decode('latin1').split('\t')[0]);
    f_val.close()

    # read train
    train_data = []
    train_label = []
    f_train = open(train, 'rb')
    for line in f_train.readlines():
        train_data.append(line.strip().decode('latin1').split('\t')[1])
        train_label.append(line.strip().decode('latin1').split('\t')[0])
    f_train.close()

    # read test
    test_data = []
    test_label = []
    f_test = open(test, 'rb')
    for line in f_test.readlines():
        test_data.append(line.strip().decode('latin1').split('\t')[1])
        test_label.append(line.strip().decode('latin1').split('\t')[0])
    f_test.close()

    import random
    random.seed(fold)
    c = list(zip(train_data, train_label)); random.shuffle(c); train_data, train_label = zip(*c)
    #c = list(zip(val_data, val_label)); random.shuffle(c); val_data, val_label = zip(*c)
    #c = list(zip(test, test_label)); random.shuffle(c); test, test_label = zip(*c)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_target = 1-le.fit_transform(train_label)
    val_target = 1-le.transform(val_label)
    test_target = 1-le.transform(test_label)
    from sklearn.utils import Bunch

    Tr = Bunch(data = train_data, target = train_target, target_names=le.classes_)
    Val = Bunch(data=val_data, target=val_target, target_names=le.classes_)
    Te = Bunch(data = test_data, target = test_target, target_names=le.classes_)

    return Tr, Val, Te

def get_datasets_Radreport_Findings_tvt(fold):
    
    #filenames
    #trainind = './data/Radreport_Findings/Radreport_Finding.train.index'
    #testind = './data/Radreport_Findings/Radreport_Finding.test.index'
    #split_labels = './data/Radreport_Findings/Radreport_Finding.txt'
    #cleaneddata = './data/Radreport_Findings/Radreport_Finding.clean.txt'
    #validation = './data/Radreport_Findings/validation.txt'

    train = './data/Radreport_Findings/train.txt'
    validation = './data/Radreport_Findings/validation_' + str(fold) + '.txt'
    test = './data/Radreport_Findings/test_' + str(fold) + '.txt'

    # read validation
    val_data = [];
    val_label = []
    f_val = open(validation, 'rb')
    for line in f_val.readlines():
        val_data.append(line.strip().decode('latin1').split('\t')[1]);
        val_label.append(line.strip().decode('latin1').split('\t')[0]);
    f_val.close()

    # read train
    train_data = []
    train_label = []
    f_train = open(train, 'rb')
    for line in f_train.readlines():
        train_data.append(line.strip().decode('latin1').split('\t')[1])
        train_label.append(line.strip().decode('latin1').split('\t')[0])
    f_train.close()

    # read test
    test_data = []
    test_label = []
    f_test = open(test, 'rb')
    for line in f_test.readlines():
        test_data.append(line.strip().decode('latin1').split('\t')[1])
        test_label.append(line.strip().decode('latin1').split('\t')[0])
    f_test.close()

    import random
    random.seed(fold)
    c = list(zip(train_data, train_label)); random.shuffle(c); train_data, train_label = zip(*c)
    #c = list(zip(val_data, val_label)); random.shuffle(c); val_data, val_label = zip(*c)
    #c = list(zip(test, test_label)); random.shuffle(c); test, test_label = zip(*c)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_target = 1-le.fit_transform(train_label)
    val_target = 1-le.transform(val_label)
    test_target = 1-le.transform(test_label)
    from sklearn.utils import Bunch

    Tr = Bunch(data = train_data, target = train_target, target_names=le.classes_)
    Val = Bunch(data=val_data, target=val_target, target_names=le.classes_)
    Te = Bunch(data = test_data, target = test_target, target_names=le.classes_)

    return Tr, Val, Te

def read_dataset_impression(seed):
    # filenames
    train = './Impression/Train/train.txt'
    validation = './Impression/Validation/validation.txt'
    test = './Impression/Test/test.txt'

    print(train)
    print(validation)
    print(test)
    
    #read train files
    train_set = []
    train_labels = []
    f = open(train)
    for line in f.readlines():
        label, report = line.split('\t');
        train_labels.append(label);  train_set.append(report);

    # read train files
    val_set = []
    val_labels = []
    f = open(validation)
    for line in f.readlines():
        label, report = line.split('\t');
        val_labels.append(label);
        val_set.append(report);

    # read train files
    test_set = []
    test_labels = []
    f = open(test)
    for line in f.readlines():
        label, report = line.split('\t');
        test_labels.append(label);
        test_set.append(report);

    f.close()

    import random
    random.seed(seed)
    c = list(zip(train_set, train_labels)); random.shuffle(c); train_set, train_labels = zip(*c)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_target = 1-le.fit_transform(train_labels)
    mapping = dict(zip(le.classes_, range(0, len(le.classes_))))
    print('Mapping of training/validation/test:')
    print(mapping)
    val_target = 1-le.transform(val_labels)
    test_target = 1-le.transform(test_labels)
    from sklearn.utils import Bunch

    Tr = Bunch(data=train_set, target=train_target, target_names=le.classes_)
    Val = Bunch(data=val_set, target=val_target, target_names=le.classes_)
    Te = Bunch(data=test_set, target=test_target, target_names=le.classes_)

    return Tr, Val, Te


def read_external_dataset_impression(file_path, seed):
    # filenames
    test = file_path

    # read train files
    test_set = []
    test_labels = []
    f = open(test)
    for line in f.readlines():
        label, report = line.split('\t');
        test_labels.append(label);
        test_set.append(report);

    f.close()

    import random
    random.seed(seed)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    test_target = 1-le.fit_transform(test_labels)
    mapping = dict(zip(le.classes_, range(0, len(le.classes_))))
    print('Mapping of evaluation set:')
    print(mapping)
    from sklearn.utils import Bunch

    Te = Bunch(data=test_set, target=test_target, target_names=le.classes_)

    return Te

def read_dataset_finding(seed):
    # filenames
    train = './Finding/Train/train.txt'
    validation = './Finding/Validation/validation.txt'
    test = './Finding/Test/test.txt'

    print(train)
    print(validation)
    print(test)
    
    #read train files
    train_set = []
    train_labels = []
    f = open(train)
    for line in f.readlines():
        label, report = line.split('\t');
        train_labels.append(label);  train_set.append(report);

    # read train files
    val_set = []
    val_labels = []
    f = open(validation)
    for line in f.readlines():
        label, report = line.split('\t');
        val_labels.append(label);
        val_set.append(report);

    # read train files
    test_set = []
    test_labels = []
    f = open(test)
    for line in f.readlines():
        label, report = line.split('\t');
        test_labels.append(label);
        test_set.append(report);

    f.close()

    import random
    random.seed(seed)
    c = list(zip(train_set, train_labels)); random.shuffle(c); train_set, train_labels = zip(*c)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_target = 1-le.fit_transform(train_labels)
    val_target = 1-le.transform(val_labels)
    test_target = 1-le.transform(test_labels)
    from sklearn.utils import Bunch

    Tr = Bunch(data=train_set, target=train_target, target_names=le.classes_)
    Val = Bunch(data=val_set, target=val_target, target_names=le.classes_)
    Te = Bunch(data=test_set, target=test_target, target_names=le.classes_)

    return Tr, Val, Te


def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    #x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]

def load_data_and_labels(positive_data_file, negative_data_file, num_folds):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [(s.strip(), np.random.randint(0, num_folds)) for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [(s.strip(), np.random.randint(0, num_folds)) for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    #x_text = [(clean_str(sent), cv) for sent, cv in x_text]
    x_text = [clean_str(sent) for sent, cv in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def batch_iter(x, y, wordpairs, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    wordpairs_shuffle = wordpairs[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], wordpairs_shuffle[start_id:end_id], i, num_batch

def batch_iter_wo_permutation(x, y, wordpairs, batch_size=64): #without permutation
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    #indices = np.random.permutation(np.arange(data_len))
    indices = np.arange(data_len)
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    wordpairs_shuffle = wordpairs[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], wordpairs_shuffle[start_id:end_id], i, num_batch


def preprocess(positive_data_file, negative_data_file, dev_sample_percentage):
    # Load data
    print("Loading data...")
    x_text, y = load_data_and_labels(positive_data_file, negative_data_file, 10)   #np.random.seed(10)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    #vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    #x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    #np.random.seed(10)   
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = []
    for i in shuffle_indices:
        x_shuffled.append(x_text[i])
    y_shuffled = y[shuffle_indices]

    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x_text, y, x_shuffled, y_shuffled
    #print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    #print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    #make list to array
    #x_train = np.array(x_train)
    #x_dev = np.array(x_dev)
    return x_train, y_train, x_dev, y_dev, max_document_length

#for terc dataset
def split_traindev_terc(trainset, ratio):
    #a = trainset
    np.random.shuffle(trainset)
    #x_train = [clean_str(' '.join(sent.split(' ')[1:])) for sent in trainset]
    #y_train = [sent.split(' ')[0] for sent in trainset]
    label0 = []; label1 = []; label2 = []; label3 = []; label4 = []; label5 = []
    data0 = []; data1 = []; data2 = []; data3 = []; data4 = []; data5 = []
    for sent in trainset:
        label = sent.split(' ')[0]
        data = clean_str(' '.join(sent.split(' ')[1:]))
        if label == '0':
            label0.append(label); data0.append(data)
        elif label == '1':
            label1.append(label); data1.append(data)
        elif label == '2':
            label2.append(label); data2.append(data)
        elif label == '3':
            label3.append(label); data3.append(data)
        elif label == '4':
            label4.append(label); data4.append(data)
        elif label == '5':
            label5.append(label); data5.append(data)

    #split
    s0 = np.floor(len(data0) * ratio).astype(int);
    s1 = np.floor(len(data1) * ratio).astype(int);
    s2 = np.floor(len(data2) * ratio).astype(int);
    s3 = np.floor(len(data3) * ratio).astype(int);
    s4 = np.floor(len(data4) * ratio).astype(int);
    s5 = np.floor(len(data5) * ratio).astype(int);
    #merge
    x_train = data0[:s0] + data1[:s1] + data2[:s2] + data3[:s3] + data4[:s4] + data5[:s5]
    y_train = label0[:s0] + label1[:s1] + label2[:s2] + label3[:s3] + label4[:s4] + label5[:s5]
    x_dev = data0[s0:] + data1[s1:] + data2[s2:] + data3[s3:] + data4[s4:] + data5[s5:]
    y_dev = label0[s0:] + label1[s1:] + label2[s2:] + label3[s3:] + label4[s4:] + label5[s5:]
    #shuffle
    shuffle = np.random.permutation(len(y_train))
    x_train = [x_train[i] for i in shuffle]
    y_train = [[y_train[i]] for i in shuffle]
    shuffle = np.random.permutation(len(y_dev))
    x_dev = [x_dev[i] for i in shuffle]
    y_dev = [[y_dev[i]] for i in shuffle]
    return x_train, y_train, x_dev, y_dev

#for 2 categories text data
def cvfolds_preprocess(positive_data_file, negative_data_file, num_folds, fold):

    print("Loading data...")
    x_text, y = load_data_and_labels(positive_data_file, negative_data_file, num_folds)     #np.random.seed(10)

    max_document_length = max([len(x.split(" ")) for x, fold in x_text])

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = []
    for i in shuffle_indices:
        x_shuffled.append(x_text[i])
    y_shuffled = y[shuffle_indices]

    x_train = []
    x_dev = []
    x_test = []
    y_train = []
    y_dev = []
    y_test = []
    for (sentence, cv), label in zip(x_shuffled, y_shuffled):
        if cv == fold:
            x_dev.append(sentence)
            y_dev.append(label)
        elif cv == (fold+1)%num_folds:
            x_test.append(sentence)
            y_test.append(label)
        else:
            x_train.append(sentence)
            y_train.append(label)
    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    y_test = np.array(y_test)
    
    del x_shuffled, y_shuffled, x_text, y

    return x_train, y_train, x_dev, y_dev, x_test, y_test, max_document_length

#for 5 categories text data
def preprocess_MedicalAbs(positive_data_file, negative_data_file, num_folds, fold):

    print("Loading data...")
    x_text, y = load_data_and_labels(positive_data_file, negative_data_file, num_folds)     #np.random.seed(10)

    max_document_length = max([len(x.split(" ")) for x, fold in x_text])

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = []
    for i in shuffle_indices:
        x_shuffled.append(x_text[i])
    y_shuffled = y[shuffle_indices]

    x_train = []
    x_dev = []
    y_train = []
    y_dev = []
    for (sentence, cv), label in zip(x_shuffled, y_shuffled):
        if cv == fold:
            x_dev.append(sentence)
            y_dev.append(label)
        else:
            x_train.append(sentence)
            y_train.append(label)
    y_train = np.array(y_train)
    y_dev = np.array(y_dev)

    del x_shuffled, y_shuffled, x_text, y

    return x_train, y_train, x_dev, y_dev, max_document_length


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


def get_doc_vec_options(dfset, word_vector_map, word_embeddings_dim, option):

    real_size = len(dfset)
    data_all_word_vec = []
    for i in range(0, real_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        #initial doc vector to get min
        doc_vec_min = np.array([math.inf for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        # initial doc vector to get max
        doc_vec_max = np.array([-math.inf for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        doc_words = dfset[i]  # one document
        if type(doc_words) is float:
            print(doc_words)
            doc_words = str(doc_words)
        
        words = doc_words.split(" ")
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
            raise ValueError('Document is Null')
            print("Document is null")
            print(i)
            print('\n')
            dfset = dfset.drop(i)

    return  data_all_word_vec


def get_doc_vec_options_google(dfset, model, word_embeddings_dim, option):

    real_size = len(dfset)
    data_all_word_vec = []
    for i in range(0, real_size):
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        #initial doc vector to get min
        doc_vec_min = np.array([math.inf for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        # initial doc vector to get max
        doc_vec_max = np.array([-math.inf for k in range(word_embeddings_dim)])  # (1, word_embeddings_dim)
        doc_words = dfset[i]  # one document
        if type(doc_words) is float:
            print(doc_words)
            doc_words = str(doc_words)
        
        words = doc_words.split(" ")
        wordcount = 0
        for word in words:
            if word in model.index2word:
                wordcount = wordcount + 1
                #contain numerical positive and negative values
                word_vector = model.vectors[model.index2word.index(word)]
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
            doc_vec = np.random.rand(300)
            data_all_word_vec.append(doc_vec)
            print("Document is null")
            print(words)
            print('\n')


    return  data_all_word_vec
