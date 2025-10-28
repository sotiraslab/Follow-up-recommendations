import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification
#from transformers.models.bert import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.metrics import classification_report, precision_recall_fscore_support
from scipy.special import softmax
np.set_printoptions(precision=5)

torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cuda:0')
print(device)

# BERT clinical
#BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"

# BERT base
#BERT_MODEL = "bert-base-uncased"

#BERT large
BERT_MODEL = "bert-large-uncased"

#Biobert large
#BERT_MODEL = "dmis-lab/biobert-large-cased-v1.1"

#Biobert base
#BERT_MODEL = "dmis-lab/biobert-base-cased-v1.1"

MAX_SEQ_LENGTH = 512

GRADIENT_ACCUMULATION_STEPS = 1
NUM_TRAIN_EPOCHS = 10
LEARNING_RATE = 1e-3
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 5
BATCH_SIZE = 16
OUTPUT_DIR = "."
MODEL_FILE_NAME = "Bert_large_impre.h5"
PATIENCE = 3000

f1_history = []
loss_history = []
no_improvement = 0

label2idx = {
    'followup': 1,
    'nofollowup': 0
}
target_names = list(label2idx.keys())

class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_inputs(example_texts, example_labels, label2idx, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""

    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label2idx[label]

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    return input_items


def get_data_loader(features, batch_size, shuffle=True):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader


def get_data(file):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    if file == 'finding':
        filepath = './s2/Finding/'
    else:
        filepath = './s2/Impression/'
    train_csv = pd.read_csv(os.path.join(filepath, 'train' + '.txt'), delimiter='\t', header=None)
    train_texts, train_labels = train_csv[1].values, train_csv[0].values
    train_features = convert_examples_to_inputs(train_texts, train_labels, label2idx, MAX_SEQ_LENGTH, tokenizer,
                                                verbose=0)
    print(len(train_csv))
    dev_csv = pd.read_csv(os.path.join(filepath, 'validation' + '.txt'), delimiter='\t', header=None)
    dev_texts, dev_labels = dev_csv[1].values, dev_csv[0].values
    dev_features = convert_examples_to_inputs(dev_texts, dev_labels, label2idx, MAX_SEQ_LENGTH, tokenizer)
    print(len(dev_csv))
    test_csv = pd.read_csv(os.path.join(filepath, 'test' + '.txt'), delimiter='\t', header=None)
    test_texts, test_labels = test_csv[1].values, test_csv[0].values
    test_features = convert_examples_to_inputs(test_texts, test_labels, label2idx, MAX_SEQ_LENGTH, tokenizer)
    print(len(test_csv))
    train_dataloader = get_data_loader(train_features, BATCH_SIZE, shuffle=True)
    dev_dataloader = get_data_loader(dev_features, BATCH_SIZE, shuffle=False)
    test_dataloader = get_data_loader(test_features, BATCH_SIZE, shuffle=False)
    return train_dataloader, dev_dataloader, test_dataloader


def train(file):
    train_dataloader, dev_dataloader, test_dataloader = get_data(file)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    model.to(device)
    num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    for _ in tqdm(range(NUM_TRAIN_EPOCHS), desc="Epoch"):
        model.train()
        tr_loss = 0
        with tqdm(train_dataloader, desc="Training iteration") as pbar:
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
                loss = outputs[0]

                if GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / GRADIENT_ACCUMULATION_STEPS

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                dev_loss, test_correct, test_predicted, _ = evaluate(model, dev_dataloader)
                f1 = f1_score(test_correct, test_predicted)

                if len(f1_history) == 0 or f1 > max(f1_history):
                    no_improvement = 0
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                else:
                    no_improvement += 1

                if no_improvement >= PATIENCE:
                    print("No improvement on development set. Finish training.")
                    break

                loss_history.append(dev_loss)
                f1_history.append(f1)

                pbar.set_postfix({'train loss': '{:.5f}'.format(loss.item()),
                                  'valid loss': '{:.3f}'.format(dev_loss),
                                  'min loss': '{:.3f}'.format(min(loss_history)),
                                  'max f1': '{:.3f}'.format(max(f1_history)),
                                  'wait': no_improvement,
                                  })
      
def evaluate(model, dataloader):
    model.eval()

    eval_loss = []
    predicted_labels, correct_labels = [], []
    predicted_probs = []

    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=input_mask,
                                          token_type_ids=segment_ids, labels=label_ids)
        # print(outputs, outputs[0])
        pred = np.argmax(outputs[1].cpu().detach().numpy(), axis=1)
        label_ids = label_ids.cpu().numpy()

        predicted_labels += list(pred)
        correct_labels += list(label_ids)
        predicted_probs += list(softmax(outputs[1].cpu().detach().numpy(), axis=1))

        eval_loss.append(outputs[0].cpu().detach().numpy())

    eval_loss = np.mean(eval_loss)

    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probs = np.array(predicted_probs)

    return eval_loss, correct_labels, predicted_labels, predicted_probs

def evalutaion_main(file):

    train_dataloader, dev_dataloader, test_dataloader = get_data(file)
    model_state_dict = torch.load(os.path.join(OUTPUT_DIR, MODEL_FILE_NAME), map_location=lambda storage, loc: storage)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, state_dict=model_state_dict,
                                                          num_labels=len(target_names))
    model.to(device)

    model.eval()

    _, train_correct, train_predicted, _ = evaluate(model, train_dataloader)
    _, dev_correct, dev_predicted, _ = evaluate(model, dev_dataloader)
    _, test_correct, test_predicted, test_probs = evaluate(model, test_dataloader)

    print("Training performance:", precision_recall_fscore_support(train_correct, train_predicted, average="binary"))
    print("Development performance:", precision_recall_fscore_support(dev_correct, dev_predicted, average="binary"))
    print("Test performance:", precision_recall_fscore_support(test_correct, test_predicted, average="binary"))
    np.save('bert_large_s2_im', test_probs)
    bert_accuracy = np.mean(test_predicted == test_correct)

    print(classification_report(test_correct, test_predicted, target_names=target_names))
    f1 = f1_score(test_correct, test_predicted)
    precision = precision_score(test_correct, test_predicted)
    recall = recall_score(test_correct, test_predicted)
    tn, fp, fn, tp = confusion_matrix(test_correct, test_predicted).ravel()
    print('acc: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1 score: {:.4f}'.format(bert_accuracy, precision, recall, f1))
    print('Confusion Matrix: tn, fp, fn, tp:', tn, fp, fn, tp)
    
if __name__ == '__main__':
    file = 'finding'
    train(file)
    evalutaion_main(file)