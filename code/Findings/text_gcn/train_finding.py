#from __future__ import division
#from __future__ import print_function
import time
import tensorflow.compat.v1 as tf
from sklearn import metrics
from utils import *
from models import GCN, MLP
import random
import os
import sys
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np

if len(sys.argv) != 2:
	sys.exit("Use: python train_finding.py <dataset>")

datasets = ['rad']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")

# Set random seed
seed = random.randint(1, 200)   #a<= seed <= b
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 100,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus_finding(
    FLAGS.dataset)
print(adj)
features = sp.identity(features.shape[0])  # featureless

print(adj.shape)
print(features.shape)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

tf.disable_eager_execution()

placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],     #####normalized symmetric adjacency matrix
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
print(features[2][1])
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels, model.prob_preb], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test), outs_val[4]

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
best_f1_val = 0
saver = tf.train.Saver()
save_dir = './Finding'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation_finding')
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,
                     model.layers[0].embedding], feed_dict=feed_dict)

    # Validation
    cost, acc, pred, labels, duration, prob_preb = evaluate(
        features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    labels_val = labels[val_mask]
    pred_val = pred[val_mask]
    acc_val = accuracy_score(labels_val, pred_val)
    prfs_val = precision_recall_fscore_support(labels_val, pred_val, average='binary')

    if prfs_val[2] > best_f1_val:
        best_f1_val = prfs_val[2]
        last_improved = epoch
        saver.save(sess=sess, save_path=save_path)
        improved_str = '*'
        # Testing
        test_cost, test_acc, pred, labels, test_duration, prob_preb = evaluate(
            features, support, y_test, test_mask, placeholders)
        prob_preb_test = prob_preb[test_mask]
        probability_preb = prob_preb_test[:, 1]
        np.save('./Finding/prob_preb_finding', probability_preb)
        labels_test = labels[test_mask]
        pred_test = pred[test_mask]
        acc_test = accuracy_score(labels_test, pred_test)
        cf = confusion_matrix(labels_test, pred_test)
        prfs_test = precision_recall_fscore_support(labels_test, pred_test, average='binary')
        print('\n\n####################################', flush=True)
        print('Accuracy for test dataset:' + str(test_acc))
        print('f1 for test dataset:' + str(prfs_test[2]))
        print('#################################### ')
        print("Binary_precision_recall_fscore_support:" + str(prfs_test))
        print('Confusion Matrix:\n', cf)

        import sys
        # Get the parent directory
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Add the parent directory to the system path
        sys.path.insert(0, parent_dir)
        
        best_epoch = epoch
        
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "f1_val", "{:.5f}".format(prfs_val[2]), "time=",
          "{:.5f}".format(time.time() - t))

    if epoch - best_epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")