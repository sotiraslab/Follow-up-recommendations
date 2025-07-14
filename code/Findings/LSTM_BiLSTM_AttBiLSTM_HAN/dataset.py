import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

label_dict = {
    'followup': 1,
    'nofollowup': 0
}

def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    text = [line.split('\t')[1] for line in lines]
    label = np.array([label_dict[line.split('\t')[0]] for line in lines])
    print(len(text), len(label))
    return text, label

def get_data(dataroot='./Finding/'):
    # read data from text file, and get text and label
    train_text, train_label = read_file(os.path.join(dataroot, 'train.txt'))
    valid_text, valid_label = read_file(os.path.join(dataroot, 'validation.txt'))
    test_text, test_label = read_file(os.path.join(dataroot, 'test.txt'))
    # get all words' idx by extending them together.
    text = []
    text.extend(train_text)


    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    tokenizer.fit_on_texts(text)
    vocab = tokenizer.word_index

    # replace work with numbers in dictionary
    X_train_word_ids = tokenizer.texts_to_sequences(train_text)
    X_valid_word_ids = tokenizer.texts_to_sequences(valid_text)
    X_test_word_ids = tokenizer.texts_to_sequences(test_text)
    # get the max length
    maxlen = np.max([np.max([len(item) for item in X_train_word_ids]),
                 np.max([len(item) for item in X_valid_word_ids]),
                 np.max([len(item) for item in X_test_word_ids])])

    # sequence, change all sequence into the same length
    x_train = pad_sequences(X_train_word_ids, maxlen=maxlen)
    x_valid = pad_sequences(X_valid_word_ids, maxlen=maxlen)
    x_test = pad_sequences(X_test_word_ids, maxlen=maxlen)
    # shuffle training set
    shuffle_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_ix]
    train_label = train_label[shuffle_ix]
    return x_train, x_valid, x_test, train_label, valid_label, test_label, vocab, maxlen

if __name__ == '__main__':
    x_train, x_valid, x_test, train_label, valid_label, test_label, vocab, maxlen = get_data()
    print(x_train.shape, train_label.shape, maxlen)