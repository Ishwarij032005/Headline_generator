# ============================================================
# Hybrid Extractive + Abstractive Headline Generator
# Seq2Seq with Attention (PyTorch)
# Author: Ishwari Jamadade
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import nltk
from datasets import load_dataset
from rouge_score import rouge_scorer

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
nltk.download("punkt", quiet=True)

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"].select(range(15000))
val_data = dataset["validation"].select(range(2000))

# ------------------------------------------------------------
# Extractive Preprocessor
# ------------------------------------------------------------
class ExtractivePreprocessor:
    def extract(self, text, max_sentences=6):
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = text.split(".")
        return " ".join(sentences[:max_sentences])

    def lead3(self, text):
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = text.split(".")
        return " ".join(sentences[:3])

extractor = ExtractivePreprocessor()

# ------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------
class HybridTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>":0, "<UNK>":1, "<SOS>":2, "<EOS>":3}
        self.idx2word = {v:k for k,v in self.word2idx.items()}

    def build_vocab(self, texts, vocab_size=35000):
        freq = Counter()
        for text in texts:
            try:
                words = nltk.word_tokenize(text.lower())
            except:
                words = text.lower().split()
            freq.update(words)

        idx = 4
        for word, _ in freq.most_common(vocab_size):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

    def encode(self, text, max_len):
        try:
            words = nltk.word_tokenize(text.lower())
        except:
            words = text.lower().split()

        ids = [self.word2idx.get(w, 1) for w in words[:max_len]]
        return ids + [0] * (max_len - len(ids))

    def decode(self, ids):
        words = []
        for i in ids:
            if i in [0, 2, 3]:
                continue
            words.append(self.idx2word.get(i, "<UNK>"))
        return " ".join(words)

tokenizer = HybridTokenizer()

articles = [train_data[i]["article"] for i in range(12000)]
highlights = [train_data[i]["highlights"] for i in range(12000)]
tokenizer.build_vocab(articles + highlights)

# ------------------------------------------------------------
# Attention Module
# ------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 4, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        return context

# ------------------------------------------------------------
# Seq2Seq Model
# ------------------------------------------------------------
class HybridSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=384):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True,
            bidirectional=True, num_layers=2, dropout=0.3
        )

        self.attention = Attention(hidden_dim)
        self.decoder = nn.LSTMCell(embed_dim + hidden_dim * 2, hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 4, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, src, trg=None, max_len=60):
        embedded = self.dropout(self.embedding(src))
        enc_out, (h, c) = self.encoder(embedded)

        h = torch.cat((h[-2], h[-1]), dim=1)
        c = torch.cat((c[-2], c[-1]), dim=1)

        outputs = []

        if trg is not None:
            trg_emb = self.dropout(self.embedding(trg))
            for t in range(trg.size(1)):
                ctx = self.attention(h, enc_out)
                inp = torch.cat((trg_emb[:, t], ctx), dim=1)
                h, c = self.decoder(inp, (h, c))
                out = self.fc(torch.cat((h, ctx), dim=1))
                outputs.append(out)
            return torch.stack(outputs, dim=1)

        else:
            inp = torch.full((src.size(0),), 2, dtype=torch.long).to(DEVICE)
            for _ in range(max_len):
                inp_emb = self.embedding(inp)
                ctx = self.attention(h, enc_out)
                dec_inp = torch.cat((inp_emb, ctx), dim=1)
                h, c = self.decoder(dec_inp, (h, c))
                out = self.fc(torch.cat((h, ctx), dim=1))
                outputs.append(out)
                inp = out.argmax(1)
            return torch.stack(outputs, dim=1)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class HeadlineDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        article = extractor.extract(item["article"])
        summary = item["highlights"]

        src = tokenizer.encode(article, 350)
        trg = tokenizer.encode(summary, 80)

        inp = [2] + trg[:-1]
        out = trg + [3]

        inp = inp[:80] + [0] * (80 - len(inp))
        out = out[:80] + [0] * (80 - len(out))

        return (
            torch.LongTensor(src),
            torch.LongTensor(inp),
            torch.LongTensor(out)
        )

# ------------------------------------------------------------
# Training Setup
# ------------------------------------------------------------
model = HybridSeq2Seq(len(tokenizer.word2idx)).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scaler = GradScaler()

train_loader = DataLoader(HeadlineDataset(train_data), batch_size=12, shuffle=True)

# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------
def train_epoch():
    model.train()
    total_loss = 0

    for src, inp, trg in train_loader:
        src, inp, trg = src.to(DEVICE), inp.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()

        with autocast():
            output = model(src, inp)
            loss = criterion(output.view(-1, output.size(-1)), trg.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# ------------------------------------------------------------
# Evaluation (ROUGE)
# ------------------------------------------------------------
def evaluate():
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = []

    for i in range(200):
        art = val_data[i]["article"]
        ref = val_data[i]["highlights"]

        src = torch.LongTensor(tokenizer.encode(extractor.extract(art), 350)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(src).argmax(2)[0].cpu().numpy()

        gen = tokenizer.decode(pred)
        score = scorer.score(ref, gen)["rouge1"].fmeasure
        scores.append(score)

    return np.mean(scores) * 100

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    for epoch in range(5):
        loss = train_epoch()
        print(f"Epoch {epoch+1}/5 | Loss: {loss:.4f}")

    rouge1 = evaluate()
    print(f"Final ROUGE-1 Score: {rouge1:.2f}")
