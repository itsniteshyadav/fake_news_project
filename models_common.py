import re
import json
from collections import Counter
import torch
import torch.nn as nn

LABELS = ["REAL", "FAKE"]

def clean_text(t: str) -> str:
    """Light cleaning. Removes Reuters source tags to prevent data leakage."""
    t = str(t)
    t = re.sub(r"^.{0,60}?\(Reuters\)\s*-?\s*", "", t)          # 'WASHINGTON (Reuters) -'
    t = re.sub(r"\bReuters\b", "", t, flags=re.I)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)                 # URLs
    t = re.sub(r"\s+", " ", t).strip()
    return t


class Vocab:
    """Word -> integer ID mapping. 0=<pad>, 1=<unk>."""
    def __init__(self, texts=None, max_size=30000, min_freq=2):
        if texts is not None:
            cnt = Counter(w for t in texts for w in self.tok(t))
            words = [w for w, c in cnt.most_common(max_size) if c >= min_freq]
            self.itos = ["<pad>", "<unk>"] + words
            self.stoi = {w: i for i, w in enumerate(self.itos)}
        else:
            self.itos = ["<pad>", "<unk>"]
            self.stoi = {"<pad>": 0, "<unk>": 1}

    @staticmethod
    def tok(t):
        return re.findall(r"[a-z0-9']+", t.lower())

    def encode(self, t, max_len=300):
        tokens = self.tok(t)
        ids = [self.stoi.get(w, 1) for w in tokens][:max_len]
        padded = ids + [0] * (max_len - len(ids))
        return padded, tokens[:max_len]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f)

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        v.itos = data["itos"]
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v


class BiLSTMAttention(nn.Module):
    """
    Embedding -> BiLSTM (reads text forwards & backwards)
              -> Self-Attention layer (computes word importance weights)
              -> Linear -> 2 classes (REAL, FAKE)
    """
    def __init__(self, vocab_size, emb=128, hid=128, layers=2, drop=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, layers, batch_first=True, bidirectional=True, dropout=drop)
        self.attn = nn.Linear(2 * hid, 1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * hid, 2)

    def forward(self, x):
        mask = (x != 0)
        h, _ = self.lstm(self.drop(self.emb(x)))                         # (B, T, 2H)
        score = self.attn(h).squeeze(-1).masked_fill(~mask, -1e9)         # (B, T)
        w = torch.softmax(score, dim=1)                                   # attention weights
        ctx = (h * w.unsqueeze(-1)).sum(dim=1)                            # context vector
        logits = self.fc(self.drop(ctx))
        return logits, w
