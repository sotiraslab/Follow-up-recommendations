#from __future__ import print_function
import gensim
import os
import sys
import time
from datetime import timedelta

import numpy as np

from dataset_hybrid import Word2VecDataset
from hybrid_generatebatch import HybridModel, WordVectors
from data_helpers import *
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc, plot_confusion_matrix, confusion_matrix
from utils import text_cleaner, convertfiletoImpression, readimpressiontoDataframe, split_train_val_test, savemodelfile,\
    loadWord2Vec, get_doc_vec, split_train_test_based_Ind, split_and_save_train_dev_test_based_Ind

flags = tf.app.flags

flags.DEFINE_string('arch', 'skip_gram', 'Architecture (skip_gram or cbow).')
flags.DEFINE_string('algm', 'negative_sampling', 'Training algorithm '
    '(negative_sampling or hierarchical_softmax).')
flags.DEFINE_integer('max_vocab_size', 0, 'Maximum vocabulary size. If > 0, '
    'the top `max_vocab_size` most frequent words are kept in vocabulary.')
flags.DEFINE_integer('min_count', 5, 'Words whose counts < `min_count` are not'
    ' included in the vocabulary.')
flags.DEFINE_float('sample', 1e-3, 'Subsampling rate.')
flags.DEFINE_integer('window_size', 5, 'Num of words on the left or right side' 
    ' of target word within a window.')

####change the embed_size and see what happened
flags.DEFINE_integer('embed_size', 300, 'Length of word vector.')
flags.DEFINE_integer('negatives', 5, 'Num of negative words to sample.')
flags.DEFINE_float('power', 0.75, 'Distortion for negative sampling.')  ###3/4
flags.DEFINE_float('alpha', 0.005, 'Initial learning rate.')
flags.DEFINE_float('min_alpha', 0.001, 'Final learning rate.')
flags.DEFINE_boolean('add_bias', True, 'Whether to add bias term to dotproduct '
    'between syn0 and syn1 vectors.')
flags.DEFINE_integer('save_per_batch', 100, 'Every `save_per_batch` batch to '
    ' tensorboard.')
flags.DEFINE_integer('print_per_batch', 100, 'print_per_batch')
flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
flags.DEFINE_string('dataset', 'Radreport', 'Used dataset') 
flags.DEFINE_string('out_dir', './textcnn_hybrid/Radreport_Finding_f1/output', 'Output directory.')

#######textcnn
# Model Hyperparameters
flags.DEFINE_integer('fold', 0, 'which fold to train')  #(0-9)
flags.DEFINE_integer('epochs', 10, 'Num of epochs to iterate training data.')
flags.DEFINE_integer('batch_size', 4, 'Batch size for input sentences.')  
flags.DEFINE_string('w2v_source', 'random', 'The source of w2v model_googlew2v (random or google)')
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_integer("balance_lambda", 0,"Parameter to balance word2vec loss.")   
flags.DEFINE_integer("balance_function", 0,"Function for dynamic balance parameter.")   
flags.DEFINE_integer("l2_reg_lambda", 1, "L2 regularization lambda (default: 0.0)")

flags.DEFINE_integer('iterations', 3000, 'The number of iteration.')

FLAGS = flags.FLAGS

save_dir = './textcnn_hybrid/Radreport_Finding_f1/checkpoints/textcnn_finding/' + FLAGS.dataset + '_' + FLAGS.w2v_source + '_epochs' + str(FLAGS.epochs) + '_bs' + str(FLAGS.batch_size) + '_bl' \
           + str(FLAGS.balance_lambda) + '_bf' + str(FLAGS.balance_function)  + '_l2' + str(FLAGS.l2_reg_lambda) + '_fold' + str(FLAGS.fold) + '_iterations' + str(FLAGS.iterations)
save_path = os.path.join(save_dir, 'best_validation')  
print(save_dir, flush=True)

def get_time_dif(start_time):
    
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def generate_sampling_table(unigram_counts, power):

    unigram_counts = np.array(unigram_counts)
    unigram_counts_1 = np.power(unigram_counts, power)
    unigram_counts_2 = np.int64(np.ceil(unigram_counts_1))

    samlping_table = []
    for i in range(len(unigram_counts_2)):
        samlping_table.extend([i] * unigram_counts_2[i])
    samlping_table = np.array(samlping_table)
    np.random.shuffle(samlping_table)

    return samlping_table

def feed_data(model, x_batch, y_batch, dropout_keep_prob, wordpairs_batch, balance_lambda):
    wordpairs_batch = np.concatenate(wordpairs_batch)
    input_words, output_words = np.hsplit(wordpairs_batch, 2)
    
    input_words = np.squeeze(input_words)
    output_words = np.squeeze(output_words)
    num_batch_wordpairs = len(wordpairs_batch)
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.dropout_keep_prob: dropout_keep_prob,
        model.balance_lambda: balance_lambda,
        model.input_words: input_words,
        model.output_words: output_words,
        model._num_batch_wordpairs: num_batch_wordpairs
    }
    return feed_dict, num_batch_wordpairs

def evaluate(model, sess, x_, y_, wordpairs_dev):
    data_len = len(x_)
    batch_eval = batch_iter_wo_permutation(x_, y_, wordpairs_dev, 256)
    total_loss = 0.0
    total_acc = 0.0
    y_preds = []
    y_trues = []
    y_probs = []
    for x_batch, y_batch, wordpairs_batch, ith_batch, num_batch  in batch_eval:
        progress =  (ith_batch + 1) /  num_batch
        learning_rate = tf.maximum(FLAGS.alpha * (1 - progress) +
                                   FLAGS.min_alpha * progress, FLAGS.min_alpha)
        batch_len = len(x_batch)
        feed_dict, num_batch_wordpairs = feed_data(model, x_batch, y_batch, 1.0, wordpairs_batch, 1.0)
        loss, acc, y_pred, prob = sess.run([model.total_losses, model.accuracy_textcnn, model.predictions, model.prob], feed_dict=feed_dict)
        y_probs.append(prob)
        y_preds.append(y_pred)
        y_trues.append(y_batch[:,1])

        total_loss += loss.mean() * batch_len
        total_acc += acc * batch_len

    y_preds = np.concatenate(y_preds, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    y_probs = np.concatenate(y_probs, axis=0)

    prfs = precision_recall_fscore_support(y_trues, y_preds, average='binary')
    cf = confusion_matrix(y_trues, y_preds)

    return total_loss / data_len, total_acc / data_len, prfs, cf, y_preds, y_trues, y_probs

def train(hybridmodel, x_train, y_train, wordpairs_train, x_dev, y_dev, wordpairs_dev, x_test, y_test, wordpairs_test):
    print("Configuring TensorBoard and Saver...", flush=True)
    tensorboard_dir = 'tensorboard/Radreport_textcnn_findings_' + FLAGS.w2v_source + '_epochs' + str(
      FLAGS.epochs) + '_bs' + str(FLAGS.batch_size) + '_bl' + str(FLAGS.balance_lambda) + '_l2' + str(FLAGS.l2_reg_lambda) + '_fold' + str(FLAGS.fold)
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("total_losses", tf.reduce_mean(hybridmodel.total_losses))
    tf.summary.scalar("accuracy", hybridmodel.accuracy_textcnn)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...", flush=True)
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  
    best_acc_val = 0.0  
    best_f1_val = 0.0
    last_improved = 0  
    require_improvement = 3000  

    flag = False
    for epoch in range(FLAGS.epochs):
        print('Epoch:', epoch + 1, flush=True)
        #print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, wordpairs_train, FLAGS.batch_size)
        for x_batch, y_batch, wordpairs_batch, ith_batch, num_batch in batch_train:
            progress = (epoch*num_batch+ith_batch + 1)/(FLAGS.epochs*num_batch)
            
            sigfunc = math.exp(-(FLAGS.balance_function * (progress - 0.5)))
            balance_lambda = FLAGS.balance_lambda*sigfunc/(sigfunc+1)
            
            feed_dict, num_batch_wordpairs = feed_data(hybridmodel, x_batch, y_batch, FLAGS.dropout_keep_prob, wordpairs_batch, balance_lambda)
            #hybridmodel._num_batch_wordpairs = num_batch_wordpairs
            #print(hybridmodel._num_batch_wordpairs)
            #hybridmodel._learning_rate = learning_rate
            if total_batch % FLAGS.save_per_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % FLAGS.print_per_batch == 0:
                feed_dict[hybridmodel.dropout_keep_prob] = 1.0
                loss_textcnn, loss_w2v, loss_train, acc_train, syn0 = session.run([hybridmodel.loss_textcnn, hybridmodel.loss_w2v, hybridmodel.total_losses, hybridmodel.accuracy_textcnn, hybridmodel.syn0], feed_dict=feed_dict)
                loss_val, acc_val, prfs, cf, y_preds, y_trues, y_probs = evaluate(hybridmodel, session, x_dev, y_dev, wordpairs_dev)  # todo

                if prfs[2] > best_f1_val:
                    best_f1_val = prfs[2]
                    
                else:
                    improved_str = ''
                
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Textcnn loss:{1:>6.2}, w2v loss:{2:>6.2}, Train Loss: {3:>6.2}, Train Acc: {4:>7.2%},' \
                      + ' Val Loss: {5:>6.2}, Val Acc: {6:>7.2%}, Time: {7} {8},' \
                      + 'Wordpairs: {9:>6}, balance_lambda:{10:>6}'
                print(msg.format(total_batch, loss_textcnn, loss_w2v.mean(), loss_train.mean(), acc_train, loss_val, acc_val, time_dif, improved_str, num_batch_wordpairs, balance_lambda), flush=True)

            feed_dict[hybridmodel.dropout_keep_prob] = FLAGS.dropout_keep_prob
            session.run(hybridmodel.grad_update_op, feed_dict=feed_dict)  
            total_batch += 1
            if total_batch - last_improved > require_improvement:
                print("No optimization for a long time, auto-stopping...", flush=True)
                flag = True
                break  
        if flag:  
            break


def main(_):

  if sys.argv[1] not in ['train', 'test']:
      raise ValueError("""usage: python run_cnn.py [train / test]""")

  # data preparation
  fold = FLAGS.fold
  Tr, Val, Te = read_dataset_finding(fold)
  x_text, y = load_data_labels(Tr)
  max_document_length = max([len(x.split(" ")) for x in x_text])

  #train
  x_train, y_train = x_text, y

  #validation
  x_dev, y_dev = load_data_labels(Val)

  # test
  x_test, y_test = load_data_labels(Te)

  #should just use training dataset!!!!!
  w2vdataset = Word2VecDataset(arch=FLAGS.arch,
                            algm=FLAGS.algm,
                            epochs=FLAGS.epochs,   ####????
                            batch_size=FLAGS.batch_size,   #####???
                            max_vocab_size=FLAGS.max_vocab_size,
                            min_count=FLAGS.min_count,
                            sample=FLAGS.sample,
                            window_size=FLAGS.window_size,
                            )

  w2vdataset.build_vocab_lists(x_train)

  print('------------1----------------', flush=True)
  if FLAGS.w2v_source == "google":
    google_news_vec = "./word2vec/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin.gz"
    model_googlew2v = gensim.models.KeyedVectors.load_word2vec_format(google_news_vec, binary=True)
    w2v2darray = w2vdataset.buildw2vmap_google(w2vdataset.table_words, model_googlew2v)
  elif FLAGS.w2v_source == "random":
    w2v2darray = np.float32(np.random.uniform( -0.5 / FLAGS.embed_size,
        0.5 / FLAGS.embed_size, [len(w2vdataset._unigram_counts), FLAGS.embed_size]))
  

  x_train, wordpairs_train = w2vdataset.get_wordpairs(x_train, FLAGS.min_count, max_document_length)
  x_dev, wordpairs_dev = w2vdataset.get_wordpairs(x_dev, FLAGS.min_count, max_document_length)
  x_test, wordpairs_test = w2vdataset.get_wordpairs(x_test, FLAGS.min_count, max_document_length)

  #sampling table for negative sampling
  #http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
  sampling_table = generate_sampling_table(w2vdataset._unigram_counts, FLAGS.power)

  print('-----------2-------------', flush=True)

  hybridmodel = HybridModel(arch=FLAGS.arch,
                           algm=FLAGS.algm,
                           embed_size=FLAGS.embed_size,
                           batch_size=FLAGS.batch_size,
                           vocab_size=len(w2vdataset._unigram_counts),
                           unigram_counts=w2vdataset._unigram_counts,
                           negatives=FLAGS.negatives,
                           sampling_table = sampling_table,
                           alpha=FLAGS.alpha,
                           min_alpha=FLAGS.min_alpha,
                           add_bias=FLAGS.add_bias,
                           sequence_length=max_document_length,
                           num_classes=y_train.shape[1],
                           filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                           num_filters=FLAGS.num_filters,
                           w2v2darray=w2v2darray,
                           l2_reg_lambda=FLAGS.l2_reg_lambda,
                           )

  print('-----------3-------------', flush=True)

  # Create target Directory if don't exist
  dirembed = os.path.join(FLAGS.out_dir, str(FLAGS.embed_size)) + '/' + FLAGS.w2v_source + '/epochs' + str(
      FLAGS.epochs) + '/bs' + str(FLAGS.batch_size) + '/bl' + str(FLAGS.balance_lambda) + '/bf' + str(FLAGS.balance_function) + '/l2' + str(FLAGS.l2_reg_lambda) + '/fold' + str(FLAGS.fold)
  print(dirembed)
  if not os.path.exists(dirembed):
      os.makedirs(dirembed)
      print("Directory ", dirembed, " Created ")
  else:
      print("Directory ", dirembed, " already exists")
  f_results = open(dirembed + '/resutls.txt', 'w')

  print('-----------4-------------', flush=True)
  if sys.argv[1] == 'train':
      print('-----------5-------------', flush=True)
      train(hybridmodel, x_train, y_train, wordpairs_train, x_dev, y_dev, wordpairs_dev, x_test, y_test, wordpairs_test)
  print('-----------6-------------', flush=True)
  with open((dirembed + '/' + 'vocab.txt'), 'w', encoding="utf-8") as fid:
    for w in w2vdataset.table_words:
      fid.write(w + '\n')
  print('Word embeddings saved to', os.path.join(dirembed + '/' 'embed.npy'), file=f_results)
  print('Vocabulary saved to', os.path.join(dirembed + '/' 'vocab.txt'), file=f_results)
  print('Vocabulary saved to', os.path.join(dirembed + '/' 'y_probs_best.npy'), file=f_results)

  embed = dirembed + '/' + 'embed.npy'
  vocab = dirembed + '/' + 'vocab.txt'
  embed = np.load(embed)
  f = open(vocab, "r")
  vocab = []
  for line in f:
      vocab.append(line.rstrip())

  WV = WordVectors(embed, vocab)
  fo = dirembed + '/' + 'model'
  savemodelfile(WV, fo, False)
  print(WV.most_similar('followup',20), flush=True)
  print(WV.most_similar('cancer',20), flush=True)
  print(WV.most_similar('right', 20), flush=True)
  print(WV.most_similar('good', 20),flush=True)
  print(WV.most_similar('negative', 20),flush=True)
  print(WV.most_similar('pulmonary', 20), flush=True)
  print(WV.most_similar('tumor', 20),flush=True)

if __name__ == '__main__':
  tf.app.run()
