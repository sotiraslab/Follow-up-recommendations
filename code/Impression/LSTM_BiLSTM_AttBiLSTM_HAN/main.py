import argparse
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from nets import TextRNN, TextBiRNN, TextAttBiRNN, HAN
from dataset import get_data

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='RNN', type=str)
parser.add_argument('--n_epochs', default=10, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--dataroot', default='./Impression/', type=str)
parser.add_argument('--max_len_sentence', default=16, type=int)


model_dic = {
    'RNN': TextRNN,
    'BiRNN': TextBiRNN,
    'AttBiRNN': TextAttBiRNN,
    'HAN': HAN
}

def main(args):
    # Load data
    x_train, x_valid, x_test, y_train, y_valid, y_test, vocab, max_len = get_data(args.dataroot)
    
    # If using HAN model, reshape the data as required
    if args.model_name == 'HAN':
        max_len_sentence = args.max_len_sentence
        max_len_word = max_len // max_len_sentence
        x_train = x_train[:, :max_len_sentence * max_len_word]
        x_valid = x_valid[:, :max_len_sentence * max_len_word]
        x_test = x_test[:, :max_len_sentence * max_len_word]
        x_train = np.reshape(x_train, [x_train.shape[0], max_len_sentence, max_len_word])
        x_valid = np.reshape(x_valid, [x_valid.shape[0], max_len_sentence, max_len_word])
        x_test = np.reshape(x_test, [x_test.shape[0], max_len_sentence, max_len_word])
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.batch_size)

    # Initialize embedding weights (random or pre-trained)
    embedding_weights = np.random.rand(len(vocab)+1, 256)  # Random initialization for embedding weights
    
    # Initialize the model with embedding weights
    model = model_dic[args.model_name](maxlen=max_len, max_features=len(vocab)+1, embedding_dims=256, embedding_weights=embedding_weights) \
        if args.model_name != 'HAN' else HAN(max_len_sentence, max_len_word, max_features=len(vocab), embedding_dims=256, embedding_weights=embedding_weights)
    
    # Define loss and optimizer
    loss_func = BinaryCrossentropy()
    optimizer = Adam(args.lr)
    
    # Early stopping setup
    patience = 3000
    best_f1 = -1000
    wait = 0
    flag = 0
    weights = None
    
    # Compute loss
    def loss(model, x, y, training):
        y_ = model(x, training=training)
        return loss_func(y_true=y, y_pred=y_)

    # Compute gradients
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    # Training loop
    for i in range(args.n_epochs):
        if flag == 1:
            break
        train_loss_avg, valid_loss_avg = tf.metrics.Mean(), tf.metrics.Mean()
        train_acc, valid_acc = tf.metrics.BinaryAccuracy(), tf.metrics.BinaryAccuracy()
        with tqdm(range(len(x_train) // args.batch_size)) as pbar:
            for i, (x_batch, y_batch) in enumerate(train_dataset):
                train_loss, grads = grad(model, x_batch, y_batch)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
                pred = []
                pred.extend(np.array(model(x_valid, training=False)).flatten())
                pred = np.array(pred).flatten()
                pred[pred < 0.5] = 0
                pred[pred >= 0.5] = 1
                f1 = f1_score(y_valid, pred)
                valid_loss = loss(model, tf.convert_to_tensor(x_valid), tf.convert_to_tensor(y_valid), False)
                train_loss_avg.update_state(train_loss)
                valid_loss_avg.update_state(valid_loss)
                train_acc.update_state(y_batch, model(x_batch, training=False))
                valid_acc.update_state(tf.convert_to_tensor(y_valid), model(tf.convert_to_tensor(x_valid), training=False))
                
                if best_f1 < f1:
                    best_f1 = f1
                    wait = 0
                    weights = model.get_weights()
                else:
                    wait += 1

                if wait >= patience:
                    print('EARLY STOPPING!!!!!!!')
                    flag = 1
                    break

                pbar.set_postfix({'train loss': '{:.5f}'.format(train_loss_avg.result()),
                                  'valid loss': '{:.5f}'.format(valid_loss),
                                  'train acc': '{:.5f}'.format(train_acc.result()),
                                  'valid acc': '{:.5f}'.format(valid_acc.result()),
                                  'wait': wait})
                pbar.update(1)

    # Set model weights to best weights
    model.set_weights(weights)

    # Evaluate on the test set
    pred = []
    for x_batch, y_batch in test_dataset:
        y_ = model(x_batch, training=False)
        pred.extend(np.array(y_).flatten())
    
    pred = np.array(pred).flatten()
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    
    print('acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1 score: {:.4f}'.format(acc, precision, recall, f1))
    print('Confusion Matrix: tn, fp, fn, tp:', tn, fp, fn, tp)
    model.save_weights('./{}_{}_{}.h5'.format('FINDING', args.model_name, 'set0'))

# Main function to execute
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)