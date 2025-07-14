import os
import random
import functools
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import evaluate

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score

from datasets import Dataset, DatasetDict
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    IntervalStrategy
)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

MAX_LEN = 4096

def make_predictions(model,df):
  # Convert summaries to a list
  sentences = df.Text.tolist()

  # Define the batch size
  batch_size = 32  # You can adjust this based on your system's memory capacity

  # Initialize an empty list to store the model outputs
  all_outputs = []

  # Process the sentences in batches
  for i in range(0, len(sentences), batch_size):
      # Get the batch of sentences
      batch_sentences = sentences[i:i + batch_size]

      # Tokenize the batch
      inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)

      # Move tensors to the device where the model is (e.g., GPU or CPU)
      inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

      # Perform inference and store the logits
      with torch.no_grad():
          outputs = model(**inputs)
          all_outputs.append(outputs['logits'])
  final_outputs = torch.cat(all_outputs, dim=0)
  #print('final_outputs:')
  #print(final_outputs.cpu())
  #print('final_outputs:')
  #print(final_outputs.cpu().numpy())
  df['final_outputs0'] = final_outputs.cpu().numpy()[:,0]
  df['final_outputs1'] = final_outputs.cpu().numpy()[:,1]
  df['predictions']=final_outputs.argmax(axis=1).cpu().numpy()
  #df_test['predictions']=df_test['predictions'].apply(lambda l:category_map[l])

  return df


def compute_metrics(eval_pred):
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def llama_preprocessing_function(examples):
    return tokenizer(examples['Text'], truncation=True, max_length=MAX_LEN)


def get_performance_metrics(df):
    
  y_test = df.target
  y_pred = df.predictions

  precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
  acc = accuracy_score(y_test, y_pred)
  dict_a = {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
  print(dict_a)
    
  print("Confusion Matrix:")
  print(confusion_matrix(y_test, y_pred))

  print("\nClassification Report:")
  print(classification_report(y_test, y_pred))

  print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
  print("Accuracy Score:", accuracy_score(y_test, y_pred))


# Adjust pandas display settings to show the full text
pd.set_option('display.max_colwidth', None)

train_finding_filepath = './Finding/Train/train.txt'
# Initialize an empty list to store the rows
traindata_finding = []

# Open and read the file
with open(train_finding_filepath, 'r') as file:
    for line in file:
        gt, text = line.split('\t')
        # Check and modify 'nofollowup'
        if gt.lower() == 'nofollowup':
            gt = 'no followup'
        # Append the data as a tuple
        traindata_finding.append((text.strip(), gt))  # .strip() to remove any trailing newlines or spaces
# Create a DataFrame from the list of tuples
df_train_finding = pd.DataFrame(traindata_finding, columns=['Text', 'GroundTruth'])
# Shuffle the DataFrame with a fixed seed
df_train_finding = df_train_finding.sample(frac=1, random_state=42).reset_index(drop=True)
# df_train_finding = df_train_finding.sample(n=10, random_state=42)
df_train_finding['target'] = df_train_finding['GroundTruth'].map({'followup': 1, 'no followup': 0})
# Display the resulting DataFrame
print(df_train_finding.head(), flush=True)


# load finding one by one
val_finding_filepath = './Finding/Validation/validation.txt'
# Initialize an empty list to store the rows
valdata_finding = []

# Open and read the file
with open(val_finding_filepath, 'r') as file:
    for line in file:
        gt, text = line.split('\t')
        # Check and modify 'nofollowup'
        if gt.lower() == 'nofollowup':
            gt = 'no followup'
        # Append the data as a tuple
        valdata_finding.append((text.strip(), gt))  # .strip() to remove any trailing newlines or spaces
# Create a DataFrame from the list of tuples
df_val_finding = pd.DataFrame(valdata_finding, columns=['Text', 'GroundTruth'])
# Shuffle the DataFrame with a fixed seed
df_val_finding = df_val_finding.sample(frac=1, random_state=42).reset_index(drop=True)
# df_val_finding = df_val_finding.sample(n=10, random_state=42)
df_val_finding['target'] = df_val_finding['GroundTruth'].map({'followup': 1, 'no followup': 0})
# Display the resulting DataFrame
print(df_val_finding.head(), flush=True)


# load finding one by one
test_finding_filepath = './Finding/Test/test.txt'
# Initialize an empty list to store the rows
testdata_finding = []

# Open and read the file
with open(test_finding_filepath, 'r') as file:
    for line in file:
        gt, text = line.split('\t')
        # Check and modify 'nofollowup'
        if gt.lower() == 'nofollowup':
            gt = 'no followup'
        # Append the data as a tuple
        testdata_finding.append((text.strip(), gt))  # .strip() to remove any trailing newlines or spaces
# Create a DataFrame from the list of tuples
df_test_finding = pd.DataFrame(testdata_finding, columns=['Text', 'GroundTruth'])
# Shuffle the DataFrame with a fixed seed
#df_test_finding = df_test_finding.sample(frac=1, random_state=42).reset_index(drop=True)
#df_test_finding = df_test_finding.sample(n=100, random_state=42)
df_test_finding['target'] = df_test_finding['GroundTruth'].map({'followup': 1, 'no followup': 0})
# Display the resulting DataFrame
print(df_test_finding.head(), flush=True)


# Converting pandas DataFrames into Hugging Face Dataset objects:
dataset_train = Dataset.from_pandas(df_train_finding.drop('GroundTruth',axis=1))
dataset_val = Dataset.from_pandas(df_val_finding.drop('GroundTruth',axis=1))
dataset_test = Dataset.from_pandas(df_test_finding.drop('GroundTruth',axis=1))


# Combine them into a single DatasetDict
dataset = DatasetDict({
    'train': dataset_train,
    'val': dataset_val,
    'test': dataset_test
})
print(dataset, flush=True)
print(dataset['train'], flush=True)
print(df_train_finding.target.value_counts(normalize=True), flush=True)


class_weights=(1/df_train_finding.target.value_counts(normalize=True).sort_index()).tolist()
class_weights=torch.tensor(class_weights)
class_weights=class_weights/class_weights.sum()
print(class_weights, flush=True)


model_name = "./LLAMA3/llama3-main/models/Meta-Llama-3-8B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
)

num_rank = 128
print('num_rank: ' + str(num_rank))

lora_config = LoraConfig(
    r = num_rank, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS'
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=2
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
print(model, flush=True)


tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1


sentences = df_test_finding.Text.tolist()
print(sentences[0:2], flush=True)

col_to_delete = ['Text']

tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
tokenized_datasets = tokenized_datasets.rename_column("target", "labels")
tokenized_datasets.set_format("torch")

collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure label_weights is a tensor
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels and convert them to long type for cross_entropy
        labels = inputs.pop("labels").long()

        # Forward pass
        outputs = model(**inputs)

        # Extract logits assuming they are directly outputted by the model
        logits = outputs.get('logits')

        # Compute custom loss with class weights for imbalanced data handling
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss

root_dir = './LLAMA3-finetune/incidentalfinding_finetune_finding_r' + str(num_rank)

print('root_dir:' + str(root_dir))

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

training_args = TrainingArguments(
    output_dir = root_dir,
    learning_rate = 1e-4,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 32,
    num_train_epochs = 10,
    weight_decay = 0.01,
    evaluation_strategy = 'steps',
    eval_steps = 25,
    save_total_limit=2,
    #save_strategy = 'epoch',
    save_steps = 25,
    save_strategy = 'steps',
    metric_for_best_model = 'f1',
    load_best_model_at_end = True
)

trainer = CustomTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['val'],
    tokenizer = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
    #class_weights=class_weights,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
)

train_result = trainer.train()
eval_result = trainer.evaluate()

print(train_result, flush=True)
print(eval_result, flush=True)

df_val_finding = make_predictions(model,df_val_finding)
df_test_finding = make_predictions(model,df_test_finding)

####save pandas
# Save the DataFrame to an Excel file
df_val_finding.to_excel(root_dir + '/val_finding.xlsx', index=False)
df_test_finding.to_excel(root_dir + '/test_finding.xlsx', index=False)


#print(df_val_finding)
#print(df_test_finding)
print('performance on validation set:')
get_performance_metrics(df_val_finding)
print('\n\n\n=====================\n\n\n')
print('performance on test set')
get_performance_metrics(df_test_finding)

metrics = train_result.metrics
max_train_samples = len(dataset_train)
metrics["train_samples"] = min(max_train_samples, len(dataset_train))
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

trainer.save_model(root_dir + "/saved_model")
