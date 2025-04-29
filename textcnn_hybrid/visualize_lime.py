import gensim
import os
import sys
import time
from datetime import timedelta
from lime.lime_text import LimeTextExplainer

import numpy as np
import json
from dataset_hybrid import Word2VecDataset
from hybrid_generatebatch import HybridModel, WordVectors
from data_helpers import *
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc, plot_confusion_matrix, confusion_matrix
from utils import text_cleaner, convertfiletoImpression, readimpressiontoDataframe, split_train_val_test, savemodelfile,\
    loadWord2Vec, get_doc_vec, split_train_test_based_Ind, split_and_save_train_dev_test_based_Ind
from collections import Counter

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
flags.DEFINE_integer('print_per_batch', 100, '每多少轮次输出在训练集和验证集上的性能')
flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
flags.DEFINE_string('dataset', 'Radreport', 'Used dataset')  #[R8, 20ng]
flags.DEFINE_string('out_dir', './data/Radreport_Impression/output', 'Output directory.')

#######textcnn
# Model Hyperparameters
flags.DEFINE_integer('fold', 0, 'which fold to train')  #(0-9)
flags.DEFINE_integer('epochs', 5, 'Num of epochs to iterate training data.')
flags.DEFINE_integer('batch_size', 16, 'Batch size for input sentences.')  ####real batch sentence (10) * wordpairs
flags.DEFINE_string('w2v_source', 'google', 'The source of w2v model_googlew2v (random or google)')
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_integer("balance_lambda", 100,"Parameter to balance word2vec loss.")   ####should be dynamic lambad, function (10, 50, 100, 500, 1000)
flags.DEFINE_integer("balance_function", 10,"Function for dynamic balance parameter.")   #### (1, 5, 10, 15, 20, 25, 30)
flags.DEFINE_integer("l2_reg_lambda", 1, "L2 regularization lambda (default: 0.0)")


FLAGS = flags.FLAGS

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
        #samlping_rate.append(element)
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

def feed_data_restore(x_batch, y_batch, dropout_keep_prob, wordpairs_batch, balance_lambda):
    
    wordpairs_batch = np.concatenate(wordpairs_batch)
    input_words, output_words = np.hsplit(wordpairs_batch, 2)
    input_words = np.squeeze(input_words)
    output_words = np.squeeze(output_words)
    num_batch_wordpairs = len(wordpairs_batch)
    feed_dict = {
        'input_x:0': x_batch,
        'input_y:0': y_batch,
        'dropout_keep_prob:0': dropout_keep_prob,
        'balance_lambda:0': balance_lambda,
        'input_word:0': input_words,
        'output_word:0': output_words,
        'Placeholder:0': num_batch_wordpairs
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


def main(_):

    sess = tf.Session()
    saver = tf.train.import_meta_graph(
          './Radreport_Impression_f1/checkpoints/textcnn_impression/Radreport_random_epochs10_bs16_bl0_bf0_l21_fold2_iterations3000/best_validation.meta')
    saver.restore(sess, tf.train.latest_checkpoint(
          './Radreport_Impression_f1/checkpoints/textcnn_impression/Radreport_random_epochs10_bs16_bl0_bf0_l21_fold2_iterations3000/'))
    all_vars = tf.get_collection('variables')

    max_document_length = 501

    x_test = ['no pulmonary embolism . peribronchial thickening mucus in bronchi and ground glass opacities consistent with aspiration . unchanged large multinodular thyroid goiter . new mm left lower lobe pulmonary nodule . recommend followup ct in approximately months .',
            'no prior studies available for comparison . there is mild atelectasis in the right lung base . lungs are otherwise clear without focal pneumonic consolidation effusion or pneumothorax . heart size and mediastinal contours are normal .',
             '1.  Slight widening of the upper mediastinum which may be related to the AP technique.  Recommend PA and lateral chest radiographs if the patient can tolerate.  If there is concern for acute aortic pathology, CT angiogram of the chest would be indicated. 2.  No acute fracture in the pelvis. 3.  1.5 cm pulmonary nodule in the left lung apex is new from 2008.  Recommend nonemergent chest CT for further characterization.  Recommend follow up of the Incidental lung nodule Additional Imaging In 1 Month with chest CT.',
             '1. Fracture of the anterior inferior aspect of the T6 vertebral body and superior endplate of T7 with adjacent hematoma. 2. No evidence of acute intra-abdominal visceral injury. 3. Indeterminate 4 to 7 mm pulmonary nodules. If no prior studies are available for comparison to document stability, recommend a 6 to 12 month followup chest CT if the patient is low risk for malignancy. If the patient is high risk for malignancy, the initial followup interval would be 3 to 6 months according to the Fleischner criteria.',
             'Correlation is made to the CT chest examinations dated 11/20/2013, 06/15/2012, and 08/27/2011. Also compared to the chest radiographs from 08/27/2011.',
             'CHEST 05/17/2015 at 22:00:  Frontal view of the chest is obtained. Comparison is made to the prior examination of 05/16/2015. Interval extubation and removal of the nasogastric tube. Interval advancement of the Swan-Ganz catheter with the distal tip overlying the left pulmonary artery. The left atrial appendage clip and intra-aortic balloon pump are again noted.  The mediastinal drains are again noted.  There is mild left basilar atelectasis. Right lung is clear. No pneumothorax. Cardiomediastinal silhouette is unchanged.']

    #should just use training dataset!!!!!
    w2vdataset = Word2VecDataset(arch=FLAGS.arch,
                            algm=FLAGS.algm,
                            epochs=FLAGS.epochs,
                            batch_size=FLAGS.batch_size,
                            max_vocab_size=FLAGS.max_vocab_size,
                            min_count=FLAGS.min_count,
                            sample=FLAGS.sample,
                            window_size=FLAGS.window_size,
                            )

    # Load
    with open('raw_vocab.json', 'r') as f:
        data = json.load(f)
        raw_vocab = Counter(data)
    
    w2vdataset.build_vocab_counter(raw_vocab)

    for i in range(0, len(x_test), 1):
    
        def predict_proba(x_):
                      
            x_, wordpairs = w2vdataset.get_wordpairs(x_, FLAGS.min_count, max_document_length)
            
            batch_eval = batch_iter_wo_permutation(x_, np.tile([1, 0], (5000,1)), wordpairs, 256)
            total_loss = 0.0
            total_acc = 0.0
            y_preds = []
            y_trues = []
            y_probs = []
            for x_batch, y_batch, wordpairs_batch, ith_batch, num_batch in batch_eval:
                batch_len = len(x_batch)
                feed_dict, num_batch_wordpairs = feed_data_restore(x_batch, y_batch, 1.0, wordpairs_batch, 1.0)
                loss, acc, y_pred, prob = sess.run(
                    ['total_losses/add:0', 'accuracy/accuracy:0', 'output_1/predictions:0', 'loss_textcnn/Softmax:0'],
                    feed_dict=feed_dict)
                y_probs.append(prob)
                y_preds.append(y_pred)
                y_trues.append(y_batch[:, 1])

                total_loss += loss.mean() * batch_len
                total_acc += acc * batch_len

            y_probs = np.concatenate(y_probs, axis=0)
            
            return y_probs

        x_ = x_test[i]
        explainer = LimeTextExplainer(class_names=['no follow up', 'follow up'])
        exp = explainer.explain_instance(x_, predict_proba)

        exp.save_to_file('./cnn5-random' + str(i) + '.html')

if __name__ == '__main__':
    tf.app.run()
