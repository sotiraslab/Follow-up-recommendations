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
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='RNN', type=str)
parser.add_argument('--n_epochs', default=10, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--dataroot', default='./Finding/', type=str)


model_dic ={
    'RNN': TextRNN,
    'BiRNN': TextBiRNN,
    'AttBiRNN': TextAttBiRNN,
    'HAN': HAN
}

def main(args):
    x_train, x_valid, x_test, y_train, y_valid, y_test, vocab, max_len = get_data(args.dataroot)
    if args.model_name == 'HAN':
        max_len_sentence, max_len_word = 8, max_len // 8
        x_train = x_train[:, :max_len_sentence * max_len_word]
        x_valid = x_valid[:, :max_len_sentence * max_len_word]
        x_test = x_test[:, :max_len_sentence * max_len_word]
        x_train = np.reshape(x_train, [x_train.shape[0], max_len_sentence, max_len_word])
        x_valid = np.reshape(x_valid, [x_valid.shape[0], max_len_sentence, max_len_word])
        x_test = np.reshape(x_test, [x_test.shape[0], max_len_sentence, max_len_word])
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # train_dataset = train_dataset.shuffle(buffer_size=len(x_train) // 2)
    # train_dataset = train_dataset.prefetch(buffer_size=args.batch_size)
    train_dataset = train_dataset.batch(args.batch_size)
    test_dataset = test_dataset.batch(args.batch_size)
    loss_func = BinaryCrossentropy()
    optimizer = Adam(args.lr)
    # ????
    model = model_dic[args.model_name](maxlen=max_len, max_features=len(vocab)+1, embedding_dims=256) \
        if args.model_name != 'HAN' else HAN(max_len_sentence, max_len_word, max_features=len(vocab), embedding_dims=256)
    # ????,??loss function, optimizer,metrics
    # model.compile(optimizer=Adam(args.lr), loss=BinaryCrossentropy(), metrics=['accuracy'])
    patience = 3000
    best_f1 = -1000
    wait = 0
    flag = 0
    weights = None
    # compute loss
    def loss(model, x, y, training):

        y_ = model(x, training=training)

        return loss_func(y_true=y, y_pred=y_)
    # compute gradients
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    for i in range(args.n_epochs):
        if flag == 1:
            break
        train_loss_avg, valid_loss_avg = tf.metrics.Mean(), tf.metrics.Mean()
        train_acc, valid_acc = tf.metrics.BinaryAccuracy(), tf.metrics.BinaryAccuracy()
        with tqdm(range(len(x_train) // args.batch_size)) as pbar:
            for i, (x_batch, y_batch) in enumerate(train_dataset):
                train_loss, grads = grad(model, x_batch, y_batch)
                # update paramters
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                # compute valid loss to judge early stop
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
                # EARLY stop,
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
                    
                if i >= 10:
                    break
                pbar.set_postfix({'train loss': '{:.5f}'.format(train_loss_avg.result()),
                                  'valid loss': '{:.5f}'.format(valid_loss),
                                  'train acc': '{:.5f}'.format(train_acc.result()),
                                  'valid acc': '{:.5f}'.format(valid_acc.result()),
                                  'wait': wait,
                                  })
                pbar.update(1)
    # ??validation loss???????
    model.set_weights(weights)

    # ??????metrics
    pred = []
    for x_batch, y_batch in test_dataset:
        y_ = model(x_batch, training=False)
        pred.extend(np.array(y_).flatten())
    # ??output
    pred = np.array(pred).flatten()
    print(pred.shape)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    tn, fp, fn, tp  = confusion_matrix(y_test, pred).ravel()
    print('acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1 score: {:.4f}'.format(acc, precision, recall, f1))
    print('Confusion Matrix: tn, fp, fn, tp:', tn, fp, fn, tp)
    model.save_weights('./{}_{}.h5'.format(args.dataroot, args.model_name)) # ????
    # plot history
    # plt.figure()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.legend(['train loss', 'valid loss'])
    # plt.grid()
    # plt.savefig('{}_history_loss.png'.format(args.model_name), dpi=600)
    # plt.figure()
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.legend(['train accuracy', 'valid accuracy'])
    # plt.grid()
    # plt.savefig('{}_history_accuracy.png'.format(args.model_name), dpi=600)

def evaluate(model):
    _, _, x_test, _, _, y_test, vocab, max_len = get_data(args.dataroot)
    if args.model_name == 'HAN':
        max_len_sentence, max_len_word = 8, max_len // 8
        x_test = x_test[:, :max_len_sentence * max_len_word]
        x_test = np.reshape(x_test, [x_test.shape[0], max_len_sentence, max_len_word])
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(args.batch_size)

    pred = []
    for x_batch, y_batch in test_dataset:
        pred.extend(np.array(model(x_batch, training=False)).flatten())
    # ??output
    pred = np.array(pred).flatten()
    predicted_probs = pred
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    eval_f1 = f1_score(y_test, pred)

    correct_labels = np.array(y_test)
    predicted_labels = np.array(pred)
    predicted_probs = np.array(predicted_probs)
    print(len(predicted_probs))
    np.save('probs1', predicted_probs)
    return eval_f1, correct_labels, predicted_labels, predicted_probs

def get_metrics(args):
    x_train, x_valid, x_test, y_train, y_valid, y_test, vocab, max_len = get_data(args.dataroot)
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    valid_dataset = valid_dataset.batch(args.batch_size)
    test_dataset = test_dataset.batch(args.batch_size)
    embedding_weights = np.random.rand(len(vocab), 256)
    model = model_dic[args.model_name](maxlen=max_len, max_features=len(vocab), embedding_dims=256, embedding_weights=embedding_weights)
    for x_batch, y_batch in test_dataset:
        np.array(model(x_batch, training=False)).flatten()
    model.load_weights('./{}_{}.h5'.format('Finding', args.model_name))
    pred = []
    for x_batch, y_batch in test_dataset:
        pred.extend(np.array(model(x_batch, training=False)).flatten())
    # ??output
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

if __name__ == '__main__':
    args = parser.parse_args()
    #main(args)
    x_train, x_valid, x_test, y_train, y_valid, y_test, vocab, max_len = get_data(args.dataroot)
    #evaluate(model)
    get_metrics(args)