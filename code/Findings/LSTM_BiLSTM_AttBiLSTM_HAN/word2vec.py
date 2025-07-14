from __future__ import division, print_function
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding,LSTM
#from tensorflow.keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import numpy as np
import os
import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords

from dataset import read_file

EMBEDDING_DIM = 300


def lower_token(tokens):
    return [w.lower() for w in tokens]


def remove_stop_words(tokens):
    stoplist = stopwords.words('english')
    return [word for word in tokens if word not in stoplist]

def remove_punct(text):
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct


def get_data(dataroot='./Finding/'):
    # read data from text file, and get text and label
    train_text, train_label = read_file(os.path.join(dataroot, 'filtered_train.txt'))
    valid_text, valid_label = read_file(os.path.join(dataroot, 'filtered_validation.txt'))
    test_text, test_label = read_file(os.path.join(dataroot, 'filtered_test.txt'))

    # get all words' idx by extending them together.
    text = []
    text.extend(train_text)
    text.extend(valid_text)
    text.extend(test_text)

    text = [remove_punct(line) for line in text]
    tokens = [word_tokenize(sen) for sen in text]
    lower_tokens = [lower_token(token) for token in tokens]
    filtered_words = [remove_stop_words(sen) for sen in lower_tokens]
    result = [' '.join(sen) for sen in filtered_words]
    x_train = result[:len(train_text)]
    x_valid = result[len(train_text):len(train_text) + len(valid_text)]
    x_test = result[len(train_text) + len(valid_text):]

    all_training_words = [word for token in tokens for word in token]
    training_sentence_lengths = [len(tokens) for tokens in tokens]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))
    print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
    print("Max sentence length is %s" % max(training_sentence_lengths))

    return x_train, x_valid, x_test, train_label, valid_label, test_label, TRAINING_VOCAB, max(training_sentence_lengths)


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors,
                                                                               generate_missing=generate_missing))
    return list(embeddings)


def tokenize(dataroot):
    x_train, x_valid, x_test, train_label, valid_label, test_label, TRAINING_VOCAB, MAX_SEQUENCE_LENGTH = get_data(dataroot)
    word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
    word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(np.concatenate([x_train, x_valid, x_test]))
    training_sequences = tokenizer.texts_to_sequences(x_train)

    train_word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(train_word_index))

    train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    shuffle_ix = np.random.permutation(np.arange(len(x_train)))
    train_cnn_data = train_cnn_data[shuffle_ix]
    train_label = train_label[shuffle_ix]

    train_embedding_weights = np.zeros((len(train_word_index) + 1, EMBEDDING_DIM))
    for word, index in train_word_index.items():
        train_embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
    print(train_embedding_weights.shape)

    valid_sequences = tokenizer.texts_to_sequences(x_valid)
    valid_cnn_data = pad_sequences(valid_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return train_embedding_weights, train_cnn_data, valid_cnn_data, test_cnn_data, train_word_index,\
           train_label, valid_label, test_label, tokenizer.word_index, MAX_SEQUENCE_LENGTH

def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=False)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2, 3, 4, 5, 6]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model

def train():
    train_embedding_weights, train_cnn_data, valid_cnn_data, test_cnn_data,\
        train_word_index, train_label, valid_label, test_label = tokenize()
    model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index) + 1, EMBEDDING_DIM, 2)

    num_epochs = 3
    batch_size = 34
    hist = model.fit(train_cnn_data, train_label, epochs=num_epochs, validation_split=(valid_cnn_data, valid_label), \
                     shuffle=True, batch_size=batch_size)

if __name__ == '__main__':
    result = get_data()
    print(len(result))
