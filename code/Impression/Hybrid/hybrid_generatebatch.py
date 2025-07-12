import heapq

import numpy as np
import tensorflow as tf


class HybridModel(object):
  """HybridModel.
  """

  def __init__(self, arch, algm, embed_size, batch_size, vocab_size, unigram_counts, negatives, sampling_table,
               alpha, min_alpha, add_bias, sequence_length, num_classes,
       filter_sizes, num_filters, w2v2darray, l2_reg_lambda=0.0):
    """Constructor  for word2vec.

    Args:
      arch: string scalar, architecture ('skip_gram' or 'cbow').
      algm: string scalar, training algorithm ('negative_sampling' or
        'hierarchical_softmax').
      embed_size: int scalar, length of word vector.
      batch_size: int scalar, batch size.
      negatives: int scalar, num of negative words to sample.
      power: float scalar, distortion for negative sampling. 
      alpha: float scalar, initial learning rate.
      min_alpha: float scalar, final learning rate.
      add_bias: bool scalar, whether to add bias term to dotproduct 
        between syn0 and syn1 vectors.
      random_seed: int scalar, random_seed.
    """

    self._arch = arch
    self._algm = algm
    self._embed_size = embed_size
    self._batch_size = batch_size
    self._vocab_size = vocab_size
    self._unigram_counts = unigram_counts
    self._negatives = negatives
    self._sampling_table = sampling_table
    self._alpha = alpha
    self._min_alpha = min_alpha
    self._add_bias = add_bias
    #self._random_seed = random_seed

    """Constructor for textcnn
    
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    
    Args:
      sequence_length, 
      num_classes, 
      vocab_size,
      embedding_size, 
      filter_sizes, 
      num_filters, 
      l2_reg_lambda=0.0
    """
    self.sequence_length = sequence_length
    self.num_classes = num_classes
    self.filter_sizes = filter_sizes
    self.num_filters = num_filters
    #self.dropout_keep_prob = dropout_keep_prob
    self.l2_reg_lambda = l2_reg_lambda
    self.w2v2darray = w2v2darray

    # five different input datasets
    #for generative model_googlew2v
    self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
    self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
    self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    self.balance_lambda = tf.placeholder(tf.float32, name='balance_lambda') #balance generative part and discriminative part
    #for discriminative model_googlew2v (word2vec)
    self.input_words = tf.placeholder(tf.int64, [None], name='input_word')
    self.output_words = tf.placeholder(tf.int64, [None], name='output_word')
    self._num_batch_wordpairs = tf.placeholder(tf.int32)  ######size of batch
    self._learning_rate = 0.001
    self.word2vec_cnn()

  def word2vec_cnn(self):
      """Hybird: word2vec + cnn"""

      #word2vec
      #syn0, syn1, biases
      syn1_rows = (self._vocab_size if self._algm == 'negative_sampling'
                   else self._vocab_size - 1)
      with tf.variable_scope(None, 'Embedding'):
          #self._syn0 = tf.get_variable('syn0', initializer=tf.random_uniform([self._vocab_size,
          #                                                              self._embed_size], -0.5 / self._embed_size,
          #                                                             0.5 / self._embed_size,
          #                                                             seed=self._random_seed))
          self._syn0 = tf.get_variable('syn0', initializer=self.w2v2darray)
          self.syn1 = tf.get_variable('syn1', initializer=tf.random_uniform([syn1_rows,
                                                                        self._embed_size], -0.1, 0.1))
          self.biases = tf.get_variable('biases', initializer=tf.zeros([syn1_rows]))

      with tf.variable_scope(None, 'loss_w2v', [self.input_words, self.output_words, self._syn0, self.syn1, self.biases]):
          if self._algm == 'negative_sampling':
              self.loss_w2v = self._negative_sampling_loss(
                  self._unigram_counts, self.input_words, self.output_words, self._syn0, self.syn1, self.biases)
          elif self._algm == 'hierarchical_softmax':
              self.loss_w2v = self._hierarchical_softmax_loss(
                  self.input_words, self.output_words, self.syn0, self.syn1, self.biases)

      #cnn
      # Keeping track of l2 regularization loss (optional)
      l2_loss = tf.constant(0.0)

      # Embedding layer
      # with tf.variable_scope("embedding_textcnn"):
      with tf.name_scope("embedding_textcnn"):
          self.embedded_chars = tf.nn.embedding_lookup(self._syn0, self.input_x)
          self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

      # Create a convolution + maxpool layer for each filter size
      pooled_outputs = []
      for i, filter_size in enumerate(self.filter_sizes):
          with tf.name_scope("conv-maxpool-%s" % filter_size):
              # Convolution Layer
              filter_shape = [filter_size, self._embed_size, 1, self.num_filters]
              W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
              b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
              conv = tf.nn.conv2d(
                  self.embedded_chars_expanded,
                  W,
                  strides=[1, 1, 1, 1],
                  padding="VALID",
                  name="conv")
              # Apply nonlinearity
              h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
              # Maxpooling over the outputs
              pooled = tf.nn.max_pool(
                  h,
                  ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                  strides=[1, 1, 1, 1],
                  padding='VALID',
                  name="pool")
              pooled_outputs.append(pooled)

      # Combine all the pooled features
      num_filters_total = self.num_filters * len(self.filter_sizes)
      self.h_pool = tf.concat(pooled_outputs, 3)
      self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

      # Add dropout
      # with tf.variable_scope('dropout'):
      with tf.name_scope("output"):
          self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

      # Final (unnormalized) scores and predictions
      # with tf.variable_scope('output'):
      with tf.name_scope("output"):
          W = tf.get_variable(
              'W',
              shape=[num_filters_total, self.num_classes],
              initializer=tf.contrib.layers.xavier_initializer())
          b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[self.num_classes]))
          # b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
          l2_loss += tf.nn.l2_loss(W)
          l2_loss += tf.nn.l2_loss(b)
          self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
          #https://stackoverflow.com/questions/41708572/tensorflow-questions-regarding-tf-argmax-and-tf-equal
          self.predictions = tf.argmax(self.scores, 1, name="predictions")

      # Calculate mean cross-entropy loss
      # with tf.variable_scope("loss"):
      with tf.name_scope("loss_textcnn"):
          self.prob = tf.nn.softmax(self.scores)
          losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
          self.loss_textcnn = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

      with tf.name_scope("total_losses"):
          #self.total_losses = tf.math.multiply(1-self.balance_lambda, self.loss_textcnn) + tf.math.multiply(self.balance_lambda, self.loss_w2v)
          self.total_losses = self.loss_textcnn + tf.math.multiply(
              self.balance_lambda, self.loss_w2v)

      with tf.name_scope("optimize"):
          global_step = tf.train.get_or_create_global_step()
          #self.learning_rate = tf.maximum(self._alpha * (1 - 0.5) +
          #                                self._min_alpha * 0.5, self._min_alpha)
          # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
          optimizer = tf.train.AdamOptimizer(self._learning_rate)
          #tf.Print(self._learning_rate, [self._learning_rate])
          #choose which layers to train
          tvars = tf.trainable_variables()
          g_vars = tvars
          #https://www.quora.com/Is-it-possible-to-only-train-the-final-layer-of-a-Neural-Net-in-TensorFlow-that-was-already-trained
          #g_vars = [var for var in tvars if 'Embedding' not in var.name]
          self.grad_update_op = optimizer.minimize(self.total_losses, global_step=global_step, var_list=g_vars)

      # Accuracy
      # with tf.variable_scope("accuracy"):
      with tf.name_scope("accuracy"):
          self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
          self.accuracy_textcnn = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

  @property
  def syn0(self):
      return self._syn0

  def _negative_sampling_loss(
      self, unigram_counts, inputs, labels, syn0, syn1, biases):
    """Builds the loss for negative sampling.

    Args:
      unigram_counts: list of int, holding word counts. Index of each entry
        is the same as the word index into the vocabulary.
      inputs: int tensor of shape [batch_size] (skip_gram) or 
        [batch_size, 2*window_size+1] (cbow)
      labels: int tensor of shape [batch_size]
      syn0: float tensor of shape [vocab_size, embed_size], input word 
        embeddings (i.e. weights of hidden layer).
      syn1: float tensor of shape [syn1_rows, embed_size], output word
        embeddings (i.e. weights of output layer).
      biases: float tensor of shape [syn1_rows], biases added onto the logits.

    Returns:
      loss: float tensor of shape [batch_size, sample_size + 1].
    """


    num_sampled = self._num_batch_wordpairs*self._negatives
    num_sampled = tf.reshape(num_sampled, [-1])
    ind_samp = tf.random.uniform(shape=num_sampled, minval=0, maxval=len(self._sampling_table), dtype=tf.int32)
    sampled = tf.gather(tf.constant(self._sampling_table),ind_samp)

    '''
    sampled_values = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=tf.expand_dims(labels, 1),
        num_true=1,
        num_sampled=10 * self._negatives,
        unique=True,
        range_max=len(unigram_counts),
        distortion=self._power,
        unigrams=unigram_counts)
    '''

    #sampled = sampled_values.sampled_candidates
    sampled_mat = tf.reshape(sampled, [-1, self._negatives])
    inputs_syn0 = self._get_inputs_syn0(syn0, inputs) # [N, D]    #inputs: batchsize*vocabularysize   syn0:vocabularysize*hiddenlayer
    true_syn1 = tf.gather(syn1, labels) # [N, D]
    sampled_syn1 = tf.gather(syn1, sampled_mat) # [N, K, D]
    true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), 1) # [N]
    sampled_logits = tf.reduce_sum(
        tf.multiply(tf.expand_dims(inputs_syn0, 1), sampled_syn1), 2) # [N, K]

    if self._add_bias:
      true_logits += tf.gather(biases, labels)  # [N]
      sampled_logits += tf.gather(biases, sampled_mat)  # [N, K]

    true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(sampled_logits), logits=sampled_logits)
    loss = tf.concat(
        [tf.expand_dims(true_cross_entropy, 1), sampled_cross_entropy], 1)
    return loss

  def _hierarchical_softmax_loss(self, inputs, labels, syn0, syn1, biases):
    """Builds the loss for hierarchical softmax.

    Args:
      inputs: int tensor of shape [batch_size] (skip_gram) or 
        [batch_size, 2*window_size+1] (cbow)
      labels: int tensor of shape [batch_size, 2*max_depth+1]
      syn0: float tensor of shape [vocab_size, embed_size], input word 
        embeddings (i.e. weights of hidden layer).
      syn1: float tensor of shape [syn1_rows, embed_size], output word
        embeddings (i.e. weights of output layer).
      biases: float tensor of shape [syn1_rows], biases added onto the logits.

    Returns:
      loss: float tensor of shape [sum_of_code_len]
    """
    inputs_syn0_list = tf.unstack(self._get_inputs_syn0(syn0, inputs))
    codes_points_list = tf.unstack(labels)
    max_depth = (labels.shape.as_list()[1] - 1) // 2
    loss = []
    for inputs_syn0, codes_points in zip(inputs_syn0_list, codes_points_list):
      true_size = codes_points[-1]
      codes = codes_points[:true_size]
      points = codes_points[max_depth:max_depth+true_size]

      logits = tf.reduce_sum(
          tf.multiply(inputs_syn0, tf.gather(syn1, points)), 1)
      if self._add_bias:
        logits += tf.gather(biases, points)

      loss.append(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.to_float(codes), logits=logits))
    loss = tf.concat(loss, axis=0)
    return loss

  def _get_inputs_syn0(self, syn0, inputs):
    """Builds the activations of hidden layer given input words embeddings 
    `syn0` and input word indices.

    Args:
      syn0: float tensor of shape [vocab_size, embed_size]
      inputs: int tensor of shape [batch_size] (skip_gram) or 
        [batch_size, 2*window_size+1] (cbow)

    Returns:
      inputs_syn0: [batch_size, embed_size]
    """
    if self._arch == 'skip_gram':
      inputs_syn0 = tf.gather(syn0, inputs)
    else:
      inputs_syn0 = []
      contexts_list = tf.unstack(inputs)
      for contexts in contexts_list:
        context_words = contexts[:-1]
        true_size = contexts[-1]
        inputs_syn0.append(
            tf.reduce_mean(tf.gather(syn0, context_words[:true_size]), axis=0))
      inputs_syn0 = tf.stack(inputs_syn0)
    return inputs_syn0

class WordVectors(object):
  """Word vectors of trained Word2Vec model_googlew2v. Provides APIs for retrieving
  word vector, and most similar words given a query word.
  """
  def __init__(self, syn0_final, vocab):
    """Constructor.

    Args:
      syn0_final: numpy array of shape [vocab_size, embed_size], final word
        embeddings.
      vocab_words: a list of strings, holding vocabulary words.
    """
    self._syn0_final = syn0_final
    self._vocab = vocab
    self._rev_vocab = dict([(w, i) for i, w in enumerate(vocab)])

  def __contains__(self, word):
    return word in self._rev_vocab

  def __getitem__(self, word):
    return self._syn0_final[self._rev_vocab[word]]

  def most_similar(self, word, k):
    """Finds the top-k words with smallest cosine distances w.r.t `word`.

    Args:
      word: string scalar, the query word.
      k: int scalar, num of words most similar to `word`.

    Returns:
      a list of 2-tuples with word and cosine similarities.
    """
    if word not in self._rev_vocab:
      raise ValueError("Word '%s' not found in the vocabulary" % word)
    if k >= self._syn0_final.shape[0]:
      raise ValueError("k = %d greater than vocabulary size" % k)

    v0 = self._syn0_final[self._rev_vocab[word]]
    sims = np.sum(v0 * self._syn0_final, 1) / (np.linalg.norm(v0) * 
        np.linalg.norm(self._syn0_final, axis=1))

    # maintain a sliding min-heap to keep track of k+1 largest elements
    min_pq = list(zip(sims[:k+1], range(k+1)))
    heapq.heapify(min_pq)
    for i in np.arange(k + 1, len(self._vocab)):
      if sims[i] > min_pq[0][0]:
        min_pq[0] = sims[i], i
        heapq.heapify(min_pq)
    min_pq = sorted(min_pq, key=lambda p: -p[0])
    return word, [(self._vocab[i], sim) for sim, i in min_pq[1:]]

