import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import logging
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_recall_fscore_support
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os
import pickle

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

parser = argparse.ArgumentParser(description="Example script to parse arguments")
parser.add_argument('--vectorsize', type=int, default=300, help='Your name')
parser.add_argument('--epoch', type=int, default=100, help='Your age')
args = parser.parse_args()
print('vectorsize: ' + str(args.vectorsize))
print('epoch: ' + str(args.epoch))

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def prepare_documents(texts, is_labeled=True):
    """
    Prepare documents for Doc2Vec training
    
    :param texts: List of texts or tuples of (text, label)
    :param is_labeled: Whether texts come with predefined labels
    :return: List of TaggedDocument objects
    """
    if is_labeled:
        # If texts are already (text, label) tuples
        tagged_docs = [TaggedDocument(words=text.split(), tags=[label]) for text, label in texts]
    else:
        # If texts are just text, use index as tag
        tagged_docs = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]
    
    return tagged_docs

def train_paragraph_vector(train_docs, vector_size=300, epochs=100):
    """
    Train a Paragraph Vector model
    
    :param train_docs: List of TaggedDocument objects
    :param vector_size: Dimensionality of the feature vectors
    :param epochs: Number of training epochs
    :return: Trained Doc2Vec model
    """
    # Initialize the model
    model = Doc2Vec(vector_size=vector_size, 
                    window=5, 
                    min_count=5, 
                    workers=4, 
                    epochs=epochs,
                    dm=1)  # Distributed Memory model
    
    # Build vocabulary
    model.build_vocab(train_docs)
    
    # Train the model
    model.train(train_docs, 
                total_examples=model.corpus_count, 
                epochs=model.epochs)
    
    return model

def infer_vectors(model, test_docs, vector_size=100):
    """
    Infer vector representations for test documents
    
    :param model: Trained Doc2Vec model
    :param test_docs: List of test documents (either texts or (text, label) tuples)
    :param vector_size: Dimensionality of vectors
    :return: Numpy array of inferred vectors
    """
    # Prepare test documents
    if isinstance(test_docs[0], tuple):
        # If labeled, extract just the text
        test_docs = [doc[0] for doc in test_docs]
    
    # Infer vectors
    vectors = [model.infer_vector(doc.split(), epochs=20) for doc in test_docs]
    
    return np.array(vectors)

def main():
    
      
    output_path = './paragraphvec/model/'
    
    #read train set
    train_list = open("./Finding/Train/train.txt",'r')
    print(train_list)   
    train_texts = []
    train_label = []
    for line in train_list:
        label, text = line.split('\t')
        report_label = (text, label)
        train_texts.append(report_label)
        if label == 'followup':
            train_label.append(1)
        elif label == 'nofollowup':
            train_label.append(0)
    train_list.close()
    y_train = np.array(train_label)
    
    validation_list = open("./Finding/Validation/validation.txt",'r')
    print(validation_list)   
    validation_texts = []
    validation_label = []
    for line in validation_list:
        label, text = line.split('\t')
        report_label = (text, label)
        validation_texts.append(report_label)
        if label == 'followup':
            validation_label.append(1)
        elif label == 'nofollowup':
            validation_label.append(0)
    validation_list.close()
    y_validation = np.array(validation_label)
    
    test_list = open("./Finding/Test/test.txt",'r')
    print(test_list)   
    test_texts = []
    test_label = []
    for line in test_list:
        label, text = line.split('\t')
        report_label = (text, label)
        test_texts.append(report_label)
        if label == 'followup':
            test_label.append(1)
        elif label == 'nofollowup':
            test_label.append(0)
    test_list.close()
    y_test = np.array(test_label)
    

    
    # Prepare documents
    train_docs = prepare_documents(train_texts)

    epcoh = args.epoch
    vector_size = args.vectorsize
    
    print('vector_size: ' + str(vector_size))
    print('epoch: ' + str(epcoh))
    
    # Train the model
    model = train_paragraph_vector(train_docs, vector_size, epcoh)
    
    # Save the model (optional)
    model.save(output_path + 'finding/' + 'finding_' + 'paragraph_vector_model_vs' + str(vector_size) + '_epoch' + str(epcoh) + '.bin')
    
    # Infer vectors for internal and external test sets
    #train_texts = train_texts[:100]; y_train = y_train[:100]
    X_train = infer_vectors(model, train_texts)
    
    #validation_texts = validation_texts[:100]; y_validation = y_validation[:100]
    X_dev = infer_vectors(model, validation_texts)
    
    #test_texts = test_texts[:100]; y_test = y_test[:100]
    X_test = infer_vectors(model, test_texts)
    
    
    # Example of how you might use the vectors
    print("Train Vectors Shape:", X_train.shape)
    print("Validation Vectors Shape:", X_dev.shape)
    print("Test Vectors Shape:", X_test.shape)
    
    
    best_f1 = 0
    kernel = 'rbf'
    gammas = [1, 1e-1, 1e-2, 1e-3]
    Cs = [1, 10, 100]
    
    for gamma in gammas:
        for c in Cs:
            random_state = 0
            clf = svm.SVC(kernel=kernel, gamma=gamma, C=c, class_weight='balanced', probability=True, random_state=random_state)
            clf.fit(X_train, y_train)
            pred_val = clf.predict(X_dev)
            f1 = f1_score(y_validation, pred_val)
            print('ker:' + str(kernel) + ' gamma:' + str(gamma) + ' c:' + str(c), flush=True)
            print('f1:' + str(f1), flush=True)
            if f1 > best_f1:
                best_f1 = f1; optimal_ker = 'rbf'; optimal_gam = gamma; optimal_c = c; best_clf = clf
            print('*'*80)
    
    
    kernel = 'linear'
    Cs = [1, 10, 100]
    for c in Cs:
        random_state = 0
        clf = svm.SVC(kernel=kernel, C=c, class_weight='balanced', probability=True, random_state=random_state)
        clf.fit(X_train, y_train)
        pred_val = clf.predict(X_dev)
        print('ker:' + str(kernel) + ' c:' + str(c), flush = True)
        f1 = f1_score(y_validation, pred_val)
        print('f1:' + str(f1), flush = True)
        if f1 > best_f1:
            best_f1 = f1; optimal_ker = 'linear'; optimal_c = c; best_clf = clf
        print('*'*80)
    
    print('best_f1:' + str(best_f1) + '    optimal_ker:' + str(optimal_ker) + '    optimal_c:' + str(optimal_c) + '    optimal_gamma:' + str(optimal_gam))
    # print('------------probability prediction-----------')
    prob_predict = best_clf.predict_proba(X_test)  # <class 'numpy.ndarray'>  shape:(999,2)
    np.save(output_path + 'finding/' + 'paragraphvec_finding_vs' + str(vector_size) + '_epoch' + str(epcoh) + '.npy', prob_predict)
    prob_label = prob_predict[:, 1].copy()
    y_pred = prob_predict[:, 1].copy()
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1
    y_true = y_test


    #Save trained model:
    output_svm_model_path = output_path + 'finding/' + 'svm_finding_model' + str(vector_size) + '_epoch' + str(epcoh) + '.pkl'
    with open(output_svm_model_path,'wb') as f:
        pickle.dump(best_clf,f)
    
    
    print(classification_report(y_true, y_pred, digits=10))
    print()
    accuracy = accuracy_score(y_true, y_pred)
    print("accuracy:" + str(accuracy))
    prfs = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print("precision_recall_fscore_support:" + str(prfs))
    print(confusion_matrix(y_true, y_pred))
    

    
if __name__ == "__main__":
    main()