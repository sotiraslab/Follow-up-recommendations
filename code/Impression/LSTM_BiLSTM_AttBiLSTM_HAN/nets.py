from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.keras.layers import Layer, Embedding, Dense, LSTM, Bidirectional, TimeDistributed, Input


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

class HAN_Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(HAN_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

class Word_Encoder(Layer):
    def __init__(self, maxlen_word, max_features, embedding_dims, embedding_weights):
        super(Word_Encoder, self).__init__()
        self.embedding = Embedding(max_features, embedding_dims, weights=[embedding_weights], input_length=maxlen_word)
        self.lstm = Bidirectional(LSTM(128, return_sequences=True))  # LSTM or GRU
        self.att = HAN_Attention(maxlen_word)

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.att(x)
        return x

class HAN(Model):
    def __init__(self,
                 maxlen_sentence,
                 maxlen_word,
                 max_features,
                 embedding_dims,
                 embedding_weights,
                 class_num=1,
                 last_activation='sigmoid'):
        super(HAN, self).__init__()
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.embedding_weights = embedding_weights
        self.class_num = class_num
        self.last_activation = last_activation
        # Word part
        # input_word = Input(shape=(self.maxlen_word,))
        # x_word = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen_word)(input_word)
        # x_word = Bidirectional(LSTM(128, return_sequences=True))(x_word)  # LSTM or GRU
        # x_word = HAN_Attention(self.maxlen_word)(x_word)
        # model_word = Model(input_word, x_word)
        # Sentence part
        self.word_encoder_att = TimeDistributed(Word_Encoder(self.maxlen_word, self.max_features, self.embedding_dims, self.embedding_weights))
        self.sentence_encoder = Bidirectional(LSTM(128, return_sequences=True))  # LSTM or GRU
        self.sentence_att = HAN_Attention(self.maxlen_sentence)
        # Output part
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 3:
            raise ValueError('The rank of inputs of HAN must be 3, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen_sentence:
            raise ValueError('The maxlen_sentence of inputs of HAN must be %d, but now is %d' % (self.maxlen_sentence, inputs.get_shape()[1]))
        if inputs.get_shape()[2] != self.maxlen_word:
            raise ValueError('The maxlen_word of inputs of HAN must be %d, but now is %d' % (self.maxlen_word, inputs.get_shape()[2]))
        x_sentence = self.word_encoder_att(inputs)
        x_sentence = self.sentence_encoder(x_sentence)
        x_sentence = self.sentence_att(x_sentence)
        output = self.classifier(x_sentence)
        return output

class TextAttBiRNN(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 embedding_weights,
                 class_num=1,
                 last_activation='sigmoid'):
        super(TextAttBiRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims,
                                   input_length=self.maxlen, weights=[embedding_weights])
        self.bi_rnn = Bidirectional(LSTM(128, return_sequences=True))  # LSTM or GRU
        self.attention = Attention(self.maxlen)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextAttBiRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextAttBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x = self.bi_rnn(embedding)
        x = self.attention(x)
        output = self.classifier(x)
        return output

class TextRNN(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 embedding_weights,
                 class_num=1,
                 last_activation='sigmoid'):
        super(TextRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims,
                                   input_length=self.maxlen, weights=[embedding_weights])
        self.rnn = LSTM(128)  # LSTM or GRU
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x = self.rnn(embedding)
        output = self.classifier(x)
        return output

class TextBiRNN(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 embedding_weights,
                 class_num=1,
                 last_activation='sigmoid'):
        super(TextBiRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(self.max_features, self.embedding_dims,
                                   input_length=self.maxlen, weights=[embedding_weights])
        self.bi_rnn = Bidirectional(LSTM(128))  # LSTM or GRU
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextBiRNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextBiRNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        embedding = self.embedding(inputs)
        x = self.bi_rnn(embedding)
        output = self.classifier(x)
        return output


if __name__ == '__main__':
    model = TextRNN(128, 1000, 256, 1)
    model.build((None, 128))
    model.summary()