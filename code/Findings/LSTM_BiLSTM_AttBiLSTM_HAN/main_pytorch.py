import os
import argparse
import re
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from gensim.models import KeyedVectors
from util import confidence_interval, raw_count_p_r_a

# Speedups
torch.backends.cudnn.benchmark = True

# =============================
# Tokenizer Utilities
# =============================
_WORD_RE = re.compile(r"\b\w+\b")
_SENT_RE = re.compile(r"[^\n.!?]+")


def simple_sentence_tokenize(text: str):
    
    # Split into rough sentences; trims spaces
    sents = _SENT_RE.findall(text)
    
    return [s.strip() for s in sents if s.strip()]


def tokenize_and_build_vocab(texts, max_vocab_size=None):
    
    tokenized_texts = [_WORD_RE.findall(text.lower()) for text in texts]
    counter = Counter([tok for doc in tokenized_texts for tok in doc])
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.most_common(max_vocab_size))}
    sequences = [[vocab.get(tok, 0) for tok in doc] for doc in tokenized_texts]
    
    return sequences, vocab


def tokenize_with_vocab(texts, vocab):
    
    tokenized_texts = [_WORD_RE.findall(text.lower()) for text in texts]
    sequences = [[vocab.get(tok, 0) for tok in doc] for doc in tokenized_texts]
    
    return sequences


def pad_sequences(sequences, maxlen=None):
    
    if maxlen is None:
        maxlen = max((len(seq) for seq in sequences), default=0)
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        padded[i, :min(len(seq), maxlen)] = seq[-maxlen:]  # right-truncation, keep last tokens
        
    return padded, maxlen

    
# ---------- HAN tokenization ----------
def han_tokenize_and_build_vocab(texts, max_vocab_size=None):
    
    # Returns list of list of word tokens per sentence
    docs = []
    counter = Counter()
    for text in texts:
        sents = simple_sentence_tokenize(text.lower())
        sent_tokens = []
        for s in sents:
            toks = _WORD_RE.findall(s)
            sent_tokens.append(toks)
            counter.update(toks)
        docs.append(sent_tokens)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.most_common(max_vocab_size))}
    id_docs = [[[vocab.get(tok, 0) for tok in sent] for sent in doc] for doc in docs]
    
    return id_docs, vocab


def han_tokenize_with_vocab(texts, vocab):
    
    docs = []
    for text in texts:
        sents = simple_sentence_tokenize(text.lower())
        sent_ids = [[vocab.get(tok, 0) for tok in _WORD_RE.findall(s)] for s in sents]
        docs.append(sent_ids)
        
    return docs


def pad_han(docs, max_sents, max_words):
    """
    docs: list of docs, each doc is list of sentences, each sentence is list of word ids
    Output: np.array shape [N, max_sents, max_words]
    """
    N = len(docs)
    arr = np.zeros((N, max_sents, max_words), dtype=int)
    for i, doc in enumerate(docs):
        sel_sents = doc[-max_sents:]  # last sentences
        for j, sent in enumerate(sel_sents):
            if not sent:
                continue
            cut = sent[-max_words:]    # last words
            arr[i, j, :len(cut)] = cut
    return arr


# =============================
# Dataset
# =============================
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.texts[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


# =============================
# Data Loader / Reader
# =============================
def read_labeled_tsv(filename, label_map):
    with open(filename, "r") as f:
        lines = f.readlines()
    texts = [line.strip().split("\t")[1] for line in lines]
    labels = np.array([label_map[line.strip().split("\t")[0]] for line in lines], dtype=np.float32)
    return texts, labels


# =============================
# Word2Vec init helper
# =============================
def build_embedding_matrix(vocab, init_mode="random", google_bin_path="GoogleNews-vectors-negative300.bin.gz", embed_dim_default=300):
    """
    init_mode: 'google' or 'random'
    Returns (embedding_matrix or None, embed_dim)
      - 'google': np.array [V+1, D], D auto from model; OOV -> U(-0.25, 0.25); freeze recommended
      - 'random': None (use PyTorch's default init in nn.Embedding), embed_dim=embed_dim_default
    """
    if init_mode == "google":
        print("Loading pretrained Word2Vec (GoogleNews 300d)...")
        w2v = KeyedVectors.load_word2vec_format(google_bin_path, binary=True)
        embed_dim = w2v.vector_size
        emb = np.zeros((len(vocab) + 1, embed_dim), dtype=np.float32)
        for word, idx in vocab.items():
            emb[idx] = w2v[word] if word in w2v else np.random.uniform(-0.25, 0.25, embed_dim)
        del w2v
        return emb, embed_dim
    else:
        # random = no external matrix; let nn.Embedding initialize and train
        return None, embed_dim_default



# =============================
# Data pipeline (train/valid/test)
# =============================
def load_data(
    dataroot,
    model_name="TextRNN",
    max_vocab_size=None,
    w2v_init="random",
    google_bin_path="GoogleNews-vectors-negative300.bin.gz",
    fixed_maxlen_for_rnn=300,
    override_maxlen=None,
    han_max_sents=30,
    han_max_words=50,
    embed_dim_default=300
):
    """
    Padding rule:
      - TextRNN: fixed (default 450) unless override
      - TextBiRNN / TextAttBiRNN: dynamic from train unless override
      - HAN: fixed grid [han_max_sents, han_max_words]
    """
    label_map = {"followup": 1, "nofollowup": 0}

    train_texts, y_train = read_labeled_tsv(os.path.join(dataroot, "train.txt"), label_map)
    valid_texts, y_valid = read_labeled_tsv(os.path.join(dataroot, "validation.txt"), label_map)
    test_texts,  y_test  = read_labeled_tsv(os.path.join(dataroot, "test.txt"), label_map)

    # ---------- HAN branch ----------
    if model_name == "HAN":
        x_train_docs, vocab = han_tokenize_and_build_vocab(train_texts, max_vocab_size)
        x_valid_docs = han_tokenize_with_vocab(valid_texts, vocab)
        x_test_docs  = han_tokenize_with_vocab(test_texts,  vocab)

        x_train = pad_han(x_train_docs, han_max_sents, han_max_words)
        x_valid = pad_han(x_valid_docs, han_max_sents, han_max_words)
        x_test  = pad_han(x_test_docs,  han_max_sents, han_max_words)

        embedding_matrix, embed_dim = build_embedding_matrix(
            vocab, init_mode=w2v_init, google_bin_path=google_bin_path, embed_dim_default=embed_dim_default
        )
        maxlen = (han_max_sents, han_max_words)  # informational only
        return x_train, y_train, x_valid, y_valid, x_test, y_test, vocab, embedding_matrix, maxlen, embed_dim

    # ---------- RNN branches ----------
    x_train, vocab = tokenize_and_build_vocab(train_texts, max_vocab_size)
    x_valid = tokenize_with_vocab(valid_texts, vocab)
    x_test  = tokenize_with_vocab(test_texts,  vocab)

    # Decide padding length
    if override_maxlen is not None:
        pad_len = int(override_maxlen)
    else:
        pad_len = fixed_maxlen_for_rnn if model_name == "TextRNN" else None  # dynamic for others

    x_train, maxlen = pad_sequences(x_train, pad_len)
    x_valid, _ = pad_sequences(x_valid, maxlen)
    x_test, _  = pad_sequences(x_test, maxlen)

    print('model_name:' + str(model_name))

    print('maxlen:' + str(maxlen))

    
    embedding_matrix, embed_dim = build_embedding_matrix(
        vocab, init_mode=w2v_init, google_bin_path=google_bin_path, embed_dim_default=embed_dim_default
    )
    return x_train, y_train, x_valid, y_valid, x_test, y_test, vocab, embedding_matrix, maxlen, embed_dim



# ---------- External data (reuses existing vocab + padding) ----------
def load_external_data(
    external_root,
    vocab,
    model_name,
    maxlen=None,
    han_max_sents=30,
    han_max_words=50,
    external_file="external.txt"
):
    """
    Load external dataset using the *existing* vocab and padding config:
      - RNN/Att: pad to 'maxlen' from training
      - HAN: pad to [han_max_sents, han_max_words]
    Expect file format: each line:  <label>\t<text>
    Labels: 'followup' or 'nofollowup'
    """
    label_map = {"followup": 1, "nofollowup": 0}
    path = os.path.join(external_root, external_file)
    texts, y_ext = read_labeled_tsv(path, label_map)

    if model_name == "HAN":
        docs = han_tokenize_with_vocab(texts, vocab)
        x_ext = pad_han(docs, han_max_sents, han_max_words)
    else:
        seqs = tokenize_with_vocab(texts, vocab)
        x_ext, _ = pad_sequences(seqs, maxlen)

    return x_ext, y_ext


# =============================
# Attention helpers
# =============================
def masked_softmax(scores, mask, dim=-1):
    """
    scores: [N, T]
    mask:   [N, T] (1 for valid, 0 for pad) or bool
    """
    if mask.dtype != torch.float32 and mask.dtype != torch.float64:
        mask = mask.float()
    # numerical stability
    scores = scores - scores.max(dim=dim, keepdim=True).values
    exp = torch.exp(scores) * mask                    # zero out pads pre-softmax
    denom = exp.sum(dim=dim, keepdim=True).clamp_min(1e-9)
    probs = exp / denom                               # rows with all-zero mask -> zeros
    return probs


# =============================
# Models (logits only)
# =============================
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_matrix=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32),
                                                 requires_grad=not freeze_embeddings)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)     # h: [1, B, H]
        h_last = h[-1]
        h_proj = self.dropout(self.relu(self.fc1(h_last)))
        return self.fc2(h_proj)       # logits


class TextBiRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_matrix=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32),
                                                 requires_grad=not freeze_embeddings)

        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.rnn(x)      # h: [2, B, H]
        h_cat = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h_cat)         # logits


class AttentionPooling(nn.Module):
    """
    Additive attention with a trainable context vector (Yang et al., 2016).
    Robust, no in-place ops (safe for autograd).
    """
    def __init__(self, in_dim, dropout=0.0, temperature=1.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim, bias=True)
        self.context = nn.Parameter(torch.empty(in_dim))
        self.dropout = nn.Dropout(dropout)
        self.temperature = float(temperature)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.context, mean=0.0, std=1.0/np.sqrt(in_dim))

    def forward(self, H, mask):
        """
        H:    [N, T, D]
        mask: [N, T] (1/True = valid, 0/False = pad)
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()
        H_ = self.dropout(H)
        u = torch.tanh(self.proj(H_))                       # [N, T, D]
        scores = torch.matmul(u, self.context) / self.temperature  # [N, T]
        weights = masked_softmax(scores, mask.float(), dim=1)      # [N, T]
        out = torch.bmm(weights.unsqueeze(1), H).squeeze(1)        # [N, D]
        return out, weights


class TextAttBiRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_matrix=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32),
                                                 requires_grad=not freeze_embeddings)

        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0)
        self.attn = AttentionPooling(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.embedding(x)                       # [B, T, E]
        H, _ = self.rnn(emb)                          # [B, T, 2H]
        mask = (x != 0).float()                       # [B, T]
        sent_vec, _ = self.attn(H, mask)
        return self.fc(sent_vec)                      # logits

        
# ---------- HAN ----------
class HAN(nn.Module):
    
    """
    Word-level BiGRU + attention to form sentence vectors,
    then sentence-level BiGRU + attention to form doc vector.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_sents, max_words,
                 embedding_matrix=None, freeze_embeddings=False):
        super().__init__()
        self.max_sents = max_sents
        self.max_words = max_words

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32),
                                                 requires_grad=not freeze_embeddings)

        word_hidden = hidden_dim // 2
        sent_hidden = hidden_dim // 2

        self.word_rnn = nn.GRU(embed_dim, word_hidden, batch_first=True, bidirectional=True)
        self.word_attn = AttentionPooling(word_hidden * 2)

        self.sent_rnn = nn.GRU(word_hidden * 2, sent_hidden, batch_first=True, bidirectional=True)
        self.sent_attn = AttentionPooling(sent_hidden * 2)

        self.fc = nn.Linear(sent_hidden * 2, 1)

    def forward(self, x):
        
        # x: [B, S, W]
        B, S, W = x.size()

        # Word level
        x_flat = x.view(B * S, W)                 # [B*S, W]
        word_mask = (x_flat != 0).float()         # [B*S, W]
        emb = self.embedding(x_flat)              # [B*S, W, E]
        H_word, _ = self.word_rnn(emb)            # [B*S, W, 2H_w]
        sent_vecs, _ = self.word_attn(H_word, word_mask)  # [B*S, 2H_w]

        # Back to sentences
        sent_vecs = sent_vecs.view(B, S, -1)      # [B, S, 2H_w]

        # Sentence level
        sent_mask = (x.sum(dim=2) != 0).float()   # [B, S]
        H_sent, _ = self.sent_rnn(sent_vecs)      # [B, S, 2H_s]
        doc_vec, _ = self.sent_attn(H_sent, sent_mask)    # [B, 2H_s]

        return self.fc(doc_vec)                   # logits

# =============================
# Training / Validation
# =============================
def train_model(model, train_loader, valid_loader, criterion, optimizer, device, n_epochs,
                patience=3000, model_name="TextRNN", dataroot="Impression", w2v_init="random", save_dir=".", runs = '0'):
    
    best_f1 = -1.0
    wait = 0
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_{dataroot}_{w2v_init}_{runs}_best_model.pt")

    for epoch in range(n_epochs):
        # ---- Train ----
        model.train()
        train_losses = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            logits = model(x).squeeze(dim=-1)    # [batch]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            # ---- Validate ----
            #model.eval()
            y_true, y_pred = [], []
            #with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).float()
                logits = model(x).squeeze(dim=-1)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            f1 = f1_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
            print(f"Epoch {epoch+1} | Loss: {np.mean(train_losses):.4f} | Val F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                wait = 0
                torch.save(model.state_dict(), save_path)
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping!")
                    break

    return save_path

# =============================
# Evaluation
# =============================
def evaluate_model(model, data_loader, device, title="Evaluation"):
    
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            logits = model(x).squeeze(dim=-1)
            probs = torch.sigmoid(logits)
            y_prob.extend(probs.cpu().numpy())
            preds = (probs >= 0.5).float()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    confidence_interval(y_true, y_pred)
    cf = confusion_matrix(y_true, y_pred)
    raw_count_p_r_a(cf)
    
    print(f"{title} — Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    return np.array(y_prob)
    
# =============================
# Main
# =============================
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="TextRNN", type=str,
                        choices=["TextRNN", "TextBiRNN", "TextAttBiRNN", "HAN"])
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--runs", default=0, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--dataroot", default="Impression", type=str)

    # Word2Vec init mode
    parser.add_argument("--w2v_init", type=str, default="random", choices=["google", "random"],
                        help="Embedding init: 'google' for GoogleNews vectors, 'random' for PyTorch init.")
    parser.add_argument("--google_bin_path", type=str, default="GoogleNews-vectors-negative300.bin.gz",
                        help="Path to GoogleNews word2vec binary (if --w2v_init google).")
    parser.add_argument("--freeze_embeddings", action="store_true",
                        help="Freeze embedding weights (useful with --w2v_init google).")

    parser.add_argument("--max_vocab_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=300, help="Used when --w2v_init is 'random'.")
    parser.add_argument("--maxlen", type=int, default=None,
                        help="Override padding length. For TextRNN defaults to fixed if not set; others use dynamic if not set.")

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints_runs")

    # HAN-specific
    parser.add_argument("--han_max_sents", type=int, default=30)
    parser.add_argument("--han_max_words", type=int, default=50)
    # TextRNN fixed length
    parser.add_argument("--textrnn_fixed_len", type=int, default=200)

    # External evaluation
    parser.add_argument("--external_dataroot", type=str, default=None,
                        help="Path to external dataset folder (set to enable external eval).")
    parser.add_argument("--external_file", type=str, default="external.txt",
                        help="Filename inside external_dataroot (default: external.txt).")

    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Using device:", device)

    print("dataroot:", args.dataroot)

    print("w2v_init:", args.w2v_init)

    
    x_train, y_train, x_valid, y_valid, x_test, y_test, vocab, embedding_matrix, maxlen, detected_embed_dim = load_data(
        dataroot=args.dataroot,
        model_name=args.model_name,
        max_vocab_size=args.max_vocab_size,
        w2v_init=args.w2v_init,
        google_bin_path=args.google_bin_path,
        fixed_maxlen_for_rnn=args.textrnn_fixed_len,
        override_maxlen=args.maxlen,
        han_max_sents=args.han_max_sents,
        han_max_words=args.han_max_words,
        embed_dim_default=args.embed_dim
    )

    
    # Pick embedding dim: from Google vectors or user-provided default
    embed_dim = detected_embed_dim

    
    train_dataset = TextDataset(x_train, y_train)
    valid_dataset = TextDataset(x_valid, y_valid)
    test_dataset  = TextDataset(x_test,  y_test)

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True)

    
    vocab_size = len(vocab) + 1
    model_map = {
        "TextRNN": lambda: TextRNN(vocab_size, embed_dim, args.hidden_dim, embedding_matrix, freeze_embeddings=args.freeze_embeddings),
        "TextBiRNN": lambda: TextBiRNN(vocab_size, embed_dim, args.hidden_dim, embedding_matrix, freeze_embeddings=args.freeze_embeddings),
        "TextAttBiRNN": lambda: TextAttBiRNN(vocab_size, embed_dim, args.hidden_dim, embedding_matrix, freeze_embeddings=args.freeze_embeddings),
        "HAN": lambda: HAN(vocab_size, embed_dim, args.hidden_dim,
                           args.han_max_sents, args.han_max_words, embedding_matrix, freeze_embeddings=args.freeze_embeddings),
    }

    
    model = model_map[args.model_name]().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    
    # Train (saves with naming scheme)
    ckpt_path = train_model(
        model, train_loader, valid_loader, criterion, optimizer, device, args.n_epochs,
        model_name=args.model_name, dataroot=args.dataroot, w2v_init=args.w2v_init, save_dir=args.save_dir, runs = str(args.runs)
    )

    
    # Load and evaluate on internal test
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    y_prob = evaluate_model(model, test_loader, device, title="Internal Test")
    np.save(args.model_name + '_' + args.dataroot + '_' + args.w2v_init + '_' + str(args.runs) + "_prob.npy", y_prob)

    
    # Optional: External evaluation
    x_ext, y_ext = load_external_data(
        external_root='../External Evaluation/',
        vocab=vocab,
        model_name=args.model_name,
        maxlen=maxlen if isinstance(maxlen, int) else None,
        han_max_sents=args.han_max_sents,
        han_max_words=args.han_max_words,
        external_file='mimic-cxr-impression-Mohamed.txt'
    )
    ext_dataset = TextDataset(x_ext, y_ext)
    ext_loader = DataLoader(ext_dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True)
    evaluate_model(model, ext_loader, device, title="External Evaluation_Mohamed")


    # Optional: External evaluation
    x_ext, y_ext = load_external_data(
        external_root='../External Evaluation/',
        vocab=vocab,
        model_name=args.model_name,
        maxlen=maxlen if isinstance(maxlen, int) else None,
        han_max_sents=args.han_max_sents,
        han_max_words=args.han_max_words,
        external_file='mimic-cxr-impression-Arash.txt'
    )
    ext_dataset = TextDataset(x_ext, y_ext)
    ext_loader = DataLoader(ext_dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True)
    evaluate_model(model, ext_loader, device, title="External Evaluation_Arash")


    # Optional: External evaluation
    x_ext, y_ext = load_external_data(
        external_root='../Temporal Evaluation/',
        vocab=vocab,
        model_name=args.model_name,
        maxlen=maxlen if isinstance(maxlen, int) else None,
        han_max_sents=args.han_max_sents,
        han_max_words=args.han_max_words,
        external_file='TemporalEvaluation.txt'
    )
    ext_dataset = TextDataset(x_ext, y_ext)
    ext_loader = DataLoader(ext_dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True)
    evaluate_model(model, ext_loader, device, title="TemporalEvaluation_CT")



if __name__ == "__main__":
    main()
