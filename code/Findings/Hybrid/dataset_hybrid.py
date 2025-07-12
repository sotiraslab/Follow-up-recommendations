import heapq
import itertools
import collections

import numpy as np
import tensorflow as tf
import tensorflow.keras as kr

from functools import partial

OOV_ID = -1

class Word2VecDataset(object):
  """Dataset for generating matrices holding word indices to train Word2Vec models.
  """
  def __init__(self,
               arch='skip_gram',
               algm='negative_sampling',
               epochs=1,
               batch_size=32,
               max_vocab_size=0,
               min_count=5,
               sample=1e-3,
               window_size=10,
               ):
    """Constructor.

    Args:
      arch: string scalar, architecture ('skip_gram' or 'cbow').
      algm: string scalar: training algorithm ('negative_sampling' or
        'hierarchical_softmax').
      epochs: int scalar, num times the dataset is iterated.
      batch_size: int scalar, the returned tensors in `get_tensor_dict` have
        shapes [batch_size, :]. 
      max_vocab_size: int scalar, maximum vocabulary size. If > 0, the top 
        `max_vocab_size` most frequent words are kept in vocabulary.
      min_count: int scalar, words whose counts < `min_count` are not included
        in the vocabulary.
      sample: float scalar, subsampling rate.
      window_size: int scalar, num of words on the left or right side of
        target word within a window.
    """
    self._arch = arch
    self._algm = algm
    self._epochs = epochs
    self._batch_size = batch_size
    self._max_vocab_size = max_vocab_size
    self._min_count = min_count
    self._sample = sample
    self._window_size = window_size

    self._iterator_initializer = None
    self._table_words = None
    self._unigram_counts = None
    self._keep_probs = None
    self._corpus_size = None
    self._max_depth = None

    #self._max_length_text = max_length_text

  @property
  def iterator_initializer(self):
    return self._iterator_initializer

  @property
  def table_words(self):
    return self._table_words

  @property
  def unigram_counts(self):
    return self._unigram_counts

  def _build_raw_vocab(self, filenames):
    """Builds raw vocabulary.

    Args:
      filenames: list of strings, holding names of text files.

    Returns:
      raw_vocab: a list of 2-tuples holding the word (string) and count (int),
        sorted in descending order of word count. 
    """
    map_open = partial(open, encoding="utf-8")
    lines = itertools.chain(*map(map_open, filenames))
    raw_vocab = collections.Counter()
    [raw_vocab.update("[pad]".split()) for i in range(10)]     #####用来填充，使得所有的句子等长
    for line in lines:
      #raw_vocab.update(line.strip().split())
      raw_vocab.update(line.split())  #########处理方式应该 same as tf.data.Dataset
    ####raw_vocab = raw_vocab.most_common()
    ####升序
    ####raw_vocab.sort(key=lambda item: item[-1])

    if self._max_vocab_size > 0:
      raw_vocab = raw_vocab[:self._max_vocab_size]
    return raw_vocab.items()

  def build_vocab(self, filenames):
    """Builds vocabulary.

    Has the side effect of setting the following attributes:
    - table_words: list of string, holding the list of vocabulary words. Index
        of each entry is the same as the word index into the vocabulary.
    - unigram_counts: list of int, holding word counts. Index of each entry
        is the same as the word index into the vocabulary.
    - keep_probs: list of float, holding words' keep prob for subsampling.
        Index of each entry is the same as the word index into the vocabulary.
    - corpus_size: int scalar, effective corpus size.

    Args:
      filenames: list of strings, holding names of text files.
    """
    raw_vocab = self._build_raw_vocab(filenames)
    self.raw_vocab = [(w, c) for w, c in raw_vocab if c >= self._min_count]
    self._corpus_size = sum(list(zip(*self.raw_vocab))[1])

    self._table_words = []
    self._unigram_counts = []
    self._keep_probs = []
    for word, count in raw_vocab:
      frac = count / float(self._corpus_size)
      keep_prob = (np.sqrt(frac / self._sample) + 1) * (self._sample / frac)
      keep_prob = np.minimum(keep_prob, 1.0).astype(np.float32)
      self._table_words.append(word)
      self._unigram_counts.append(count)
      self._keep_probs.append(keep_prob)

  def build_vocab_lists(self, X_test):

    raw_vocab = collections.Counter()
    raw_vocab.update("[pad]".split())
    for line in X_test:
      raw_vocab.update(line.split())  
    
    # Save
    # import json
    # with open('raw_vocab.json', 'w') as f:
    #    json.dump(raw_vocab, f)
    
    if self._max_vocab_size > 0:
      raw_vocab = raw_vocab[:self._max_vocab_size]
    raw_vocab = raw_vocab.items()
    self.raw_vocab = [(w, c) for w, c in raw_vocab if c >= self._min_count]
    self._corpus_size = sum(list(zip(*raw_vocab))[1])

    self._table_words = []
    self._unigram_counts = []
    self._keep_probs = []
    for word, count in raw_vocab:
      frac = count / float(self._corpus_size)
      keep_prob = (np.sqrt(frac / self._sample) + 1) * (self._sample / frac)
      keep_prob = np.minimum(keep_prob, 1.0).astype(np.float32)
      self._table_words.append(word)
      self._unigram_counts.append(count)
      self._keep_probs.append(keep_prob)

  def build_vocab_counter(self, raw_vocab):
    
    if self._max_vocab_size > 0:
      raw_vocab = raw_vocab[:self._max_vocab_size]
    raw_vocab = raw_vocab.items()
    
    self.raw_vocab = [(w, c) for w, c in raw_vocab if c >= self._min_count]
    self._corpus_size = sum(list(zip(*raw_vocab))[1])

    self._table_words = []
    self._unigram_counts = []
    self._keep_probs = []
    for word, count in raw_vocab:
      frac = count / float(self._corpus_size)
      keep_prob = (np.sqrt(frac / self._sample) + 1) * (self._sample / frac)
      keep_prob = np.minimum(keep_prob, 1.0).astype(np.float32)
      self._table_words.append(word)
      self._unigram_counts.append(count)
      self._keep_probs.append(keep_prob)
        
  def build_word_class_correlations(self, X_train, y_train):
    """Builds vocabulary.

    Has the side effect of setting the following attributes:
    - table_words: list of string, holding the list of vocabulary words. Index
        of each entry is the same as the word index into the vocabulary.
    - unigram_counts: list of int, holding word counts. Index of each entry
        is the same as the word index into the vocabulary.
    - keep_probs: list of float, holding words' keep prob for subsampling.
        Index of each entry is the same as the word index into the vocabulary.
    - corpus_size: int scalar, effective corpus size.

    Args:
      filenames: list of strings, holding names of text files.
    """
    vocab_pos = collections.Counter()
    vocab_pos.update("[pad]".split())  #####用来填充，使得所有的句子等长
    vocab_neg = collections.Counter()
    vocab_neg.update("[pad]".split())  #####用来填充，使得所有的句子等长
    vocab_all = collections.Counter()
    vocab_all.update("[pad]".split())  #####用来填充，使得所有的句子等长
    for x, y in zip(X_train, y_train):
        # raw_vocab.update(line.strip().split())
        if y[1] == 1:
            vocab_pos.update(x.split())  #
            vocab_all.update(x.split())
        else:
            vocab_neg.update(x.split())
            vocab_all.update(x.split())
    vocab_pos = {w: c for w, c in vocab_pos.items() if c >= self._min_count}
    vocab_neg = {w: c for w, c in vocab_neg.items() if c >= self._min_count}
    vocab_all = {w: c for w, c in vocab_all.items() if c >= self._min_count}
    a = []
    for w in vocab_all:
        if w in vocab_neg and w in vocab_pos:
            a.append((w, vocab_pos[w], vocab_neg[w], vocab_pos[w]/(vocab_pos[w]+vocab_neg[w])))
        elif w in vocab_neg and w not in vocab_pos:
            a.append((w, 0, vocab_neg[w], 0))
        elif w not in vocab_neg and w in vocab_pos:
            a.append((w, vocab_pos[w], 0, 1))
    a.sort(key=lambda tup: tup[3], reverse=True)


  def _build_binary_tree(self, unigram_counts):
    """Builds a Huffman tree for hierarchical softmax. Has the side effect
    of setting `max_depth`.

    Args:
      unigram_counts: list of int, holding word counts. Index of each entry
        is the same as the word index into the vocabulary.

    Returns:
      codes_points: an int numpy array of shape [vocab_size, 2*max_depth+1]
        where each row holds the codes (0-1 binary values) padded to
        `max_depth`, and points (non-leaf node indices) padded to `max_depth`,
        of each vocabulary word. The last entry is the true length of code
        and point (<= `max_depth`).
    """
    vocab_size = len(unigram_counts)
    heap = [[unigram_counts[i], i] for i in range(vocab_size)]
    heapq.heapify(heap)
    for i in range(vocab_size - 1):
      min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
      heapq.heappush(heap, [min1[0] + min2[0], i + vocab_size, min1, min2])

    node_list = []
    max_depth, stack = 0, [[heap[0], [], []]]
    while stack:
      node, code, point = stack.pop()
      if node[1] < vocab_size:
        node.extend([code, point, len(point)])
        max_depth = np.maximum(len(code), max_depth)
        node_list.append(node)
      else:
        point = np.array(list(point) + [node[1]-vocab_size])
        stack.append([node[2], np.array(list(code)+[0]), point])
        stack.append([node[3], np.array(list(code)+[1]), point])

    node_list = sorted(node_list, key=lambda items: items[1])
    codes_points = np.zeros([vocab_size, max_depth*2+1], dtype=np.int32)
    for i in range(len(node_list)):
      length = node_list[i][4] # length of code or point
      codes_points[i, -1] = length
      codes_points[i, :length] = node_list[i][2] # code
      codes_points[i, max_depth:max_depth+length] = node_list[i][3] # point
    self._max_depth = max_depth
    return codes_points

  def _prepare_inputs_labels(self, tensor):
    """Set shape of `tensor` according to architecture and training algorithm,
    and split `tensor` into `inputs` and `labels`.

    Args:
      tensor: rank-2 int tensor, holding word indices for prediction inputs
        and prediction labels, returned by `generate_instances`.

    Returns:
      inputs: rank-2 int tensor, holding word indices for prediction inputs. 
      labels: rank-2 int tensor, holding word indices for prediction labels.
    """
    if self._arch == 'skip_gram':
      if self._algm == 'negative_sampling':
        tensor.set_shape([self._batch_size, 2])
      else:
        tensor.set_shape([self._batch_size, 2*self._max_depth+2])
      inputs = tensor[:, :1]
      labels = tensor[:, 1:]
    else:
      if self._algm == 'negative_sampling':
        tensor.set_shape([self._batch_size, 2*self._window_size+2])
      else:
        tensor.set_shape([self._batch_size,
            2*self._window_size+2*self._max_depth+2])
      inputs = tensor[:, :2*self._window_size+1]
      labels = tensor[:, 2*self._window_size+1:]
    return inputs, labels

  def get_wordpairs(self, dataset, min_count, max_document_length):

    table_words = self._table_words
    unigram_counts = self._unigram_counts
    keep_probs = self._keep_probs

    if not table_words or not unigram_counts or not keep_probs:
      raise ValueError('`table_words`, `unigram_counts`, and `keep_probs` must',
                         'be set by calling `build_vocab()`')

    if self._algm == 'hierarchical_softmax':
      codes_points = tf.constant(self._build_binary_tree(unigram_counts))
    elif self._algm == 'negative_sampling':
      codes_points = None
    else:
      raise ValueError('algm must be hierarchical_softmax or negative_sampling')

    #build a dict   (word and id)
    word_to_id = dict(zip(table_words, range(len(table_words))))

    wordpairs = generate_wordpairs(dataset, min_count, max_document_length, word_to_id, keep_probs,
                                   self._window_size, self._arch, self.raw_vocab, codes_points)

    return wordpairs

  def buildw2vmap_partgoogle(self, table_words, word_vector_map):

      dimension = len(word_vector_map['the'])
      w2v2darray = []
      #i = 0
      for word in table_words:
          if word in word_vector_map:
              word_vector = word_vector_map[word]
          else:
              #print(word)
              #i = i + 1
              word_vector = np.random.uniform( -0.5/dimension,
        0.5/dimension, dimension)
          w2v2darray.append(word_vector)
      #print(i)
      return np.float32(np.array(w2v2darray))

  def buildw2vmap_google(self, table_words, model):

      w2v2darray = []
      #i = 0
      for word in table_words:
          if word in model.index_to_key:
              word_vector = model.vectors[model.index_to_key.index(word)]
          else:
              #print(word)
              #i = i + 1
              word_vector = np.random.uniform( -0.5/300,
        0.5/300, 300)
          w2v2darray.append(word_vector)
      #print(i)
      return np.float32(np.array(w2v2darray))

  def buildw2vmap_pretrain(self, table_words, model):

      w2v2darray = []
      #i = 0
      for word in table_words:
          if word in model:
              word_vector = model[word]
          else:
              #print(word)
              #i = i + 1
              word_vector = np.random.uniform( -0.5/300,
        0.5/300, 300)
          w2v2darray.append(word_vector)
      #print(i)
      return np.float32(np.array(w2v2darray))

    
def generate_wordpairs(dataset, min_count, max_document_length, word_to_id, keep_probs, window_size, arch, raw_vocab, codes_points):

    def per_target_fn(index):
        #reduced_size = tf.random_uniform([], maxval=window_size, dtype=tf.int32)
        reduced_size = np.random.randint(0, high=window_size)
        left = list(range(np.maximum(index - window_size + reduced_size, 0), index))
        right = list(range(index + 1,
                         np.minimum(index + 1 + window_size - reduced_size, np.size(subindices))))
        context = left + right  ####这里求出来的相当于索引，还要把索引对应的元素取出来
        context = [subindices[i] for i in context]

        if arch == 'skip_gram':
            input_words = [subindices[index] for _ in range(len(context))]
            window = np.column_stack((input_words, context))
        elif arch == 'cbow':
            true_size = context.size
            # true_size = tf.Print(true_size,[true_size],summarize=20)
            # print('true_size:',true_size.shape)
            #pads = tf.pad(context, [[0, 2 * window_size - true_size]])
            # tf.Print(pads,[pads], summarize=20)
            window = tf.concat([tf.pad(context, [[0, 2 * window_size - true_size]]),
                                [true_size, indices[index]]], axis=0)
            # tf.Print(window,[window],summarize=20)
            # print('window:',window.shape)
            window = tf.expand_dims(window, axis=0)
            # tf.Print(window,[window],summarize=20)
            # print('window:', window.shape)
        else:
            raise ValueError('architecture must be skip_gram or cbow.')

        if codes_points is not None:
            window = tf.concat([window[:, :-1],
                                tf.gather(codes_points, window[:, -1])], axis=1)
        return window

    #list to dict
    raw_vocab = dict(raw_vocab)

    data_id = []
    filt_data_id = []  #min_count
    for i in range(len(dataset)):  #keep length of each sentence.
        data_id.append([word_to_id[x] for x in dataset[i].split() if x in word_to_id])
        filt_data_id.append([word_to_id[x] for x in dataset[i].split() if x in raw_vocab.keys() and raw_vocab[x] >=min_count])

    #use min_count to filter each sentence

    keep_probs = np.array(keep_probs)  #list to array
    all_windows = []
    #all_data = []
    #all_labels = []
    count = 0
    for indices, indices_full in zip(filt_data_id, data_id):
        if len(indices) < 2:
            #print(indices)
            #print(len(indices_full))
            count = count + 1
            if len(indices_full) >= 2:
                indices = indices_full
            elif len(indices_full) == 1:
                indices = 2*indices_full
            else:
                indices = 2*[0]
            #continue
        keep_probs_text = keep_probs[indices]
        randvars = np.random.uniform(0, 1, np.size(keep_probs_text))
        boolean_mask = randvars < keep_probs_text
        subindices = [b for a, b in zip(boolean_mask, indices) if a]
        if len(subindices) < 2:
            #if downsample, cannot form wordpairs, then no dowmsample
            subindices = indices
            #all_windows.append(np.array([[np.random.randint(low=1, high=len(word_to_id)-1)]*2, [np.random.randint(low=1, high=len(word_to_id)-1)]*2]))
            #all_data.append(indices)
            #all_labels.append(label)
            #print(indices)
            #print(subindices)
            #print()
            #count = count + 1
            #continue
            #raise ValueError('the size of subindices must be larger than 1')
        index = 0
        windows = []
        while index < len(subindices):
            window = per_target_fn(index)
            windows.append(window)
            index += 1
        windows = np.concatenate(windows)
        '''uiu
        if windows.shape[0] < num_wordpairs:
            windows = true_fn(windows)u
        else:
            windows = false_fn(windows)
        '''
        all_windows.append(windows)
        #all_data.append(indices)
        #all_labels.append(label)
    #print(count)
    #pad 0 to dataset
    data_id = kr.preprocessing.sequence.pad_sequences(data_id, maxlen=max_document_length, padding='post')
    #filt_data_id = kr.preprocessing.sequence.pad_sequences(filt_data_id, maxlen=max_document_length, padding='post')
    return data_id, np.array(all_windows)

