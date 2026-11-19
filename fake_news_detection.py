"""
Fake News Detection: TF-IDF baseline vs BiLSTM+Attention vs BERT
=================================================================
Dataset : ISOT Fake/Real News  (Fake.csv, True.csv)  -> kaggle: "fake-and-real-news-dataset"
          Any CSV pair with columns [title, text] works.
Labels  : 0 = REAL, 1 = FAKE

Pipeline (say this in the viva):
  1. Load + clean data (remove leakage such as "(Reuters)", drop duplicates)
  2. Stratified 70/15/15 train/val/test split
  3. Model 1: TF-IDF + Logistic Regression   (simple baseline)
  4. Model 2: BiLSTM + Attention             (sequence model, explainable)
  5. Model 3: BERT fine-tuning               (transformer, state of the art)
  6. Evaluate all on the SAME test set (accuracy, precision, recall, F1, ROC-AUC)
  7. Explainability + error analysis

Run examples:
  python fake_news_detection.py --data_dir ./data                      # all 3 models
  python fake_news_detection.py --data_dir ./data --models baseline,lstm --max_samples 5000   # quick test
  python fake_news_detection.py --data_dir ./data --models bert --bert_name distilbert-base-uncased
"""
import argparse
import os
import re
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, roc_curve)

LABELS = ["REAL", "FAKE"]


# ----------------------------------------------------------------------------
# 0. Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# 1. Data loading and cleaning
# ----------------------------------------------------------------------------
def clean_text(t: str) -> str:
    """Light cleaning. We keep it minimal because BERT works best on natural text.
    IMPORTANT: the ISOT dataset has a *data leakage* problem: real articles start with
    'CITY (Reuters) -' so a model can score ~99% by just spotting 'Reuters'.
    We remove those source tags so the model must learn writing style/content instead."""
    t = str(t)
    t = re.sub(r"^.{0,60}?\(Reuters\)\s*-?\s*", "", t)          # 'WASHINGTON (Reuters) -'
    t = re.sub(r"\bReuters\b", "", t, flags=re.I)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)                 # URLs
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_data(data_dir, max_samples=None):
    fake = pd.read_csv(os.path.join(data_dir, "Fake.csv"))
    true = pd.read_csv(os.path.join(data_dir, "True.csv"))
    fake["label"], true["label"] = 1, 0
    df = pd.concat([fake, true], ignore_index=True)

    # We use ONLY title + text. 'subject' and 'date' are dropped (they leak the label).
    df["content"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).map(clean_text)
    df = df[df["content"].str.split().str.len() > 20]            # remove near-empty rows
    df = df.drop_duplicates(subset="content")                    # duplicates would leak train->test
    df = df[["content", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)
    if max_samples:
        df = df.iloc[:max_samples]
    print(f"[data] {len(df)} articles | fake={df.label.sum()} real={(df.label == 0).sum()}")
    return df


def split_data(df):
    train, temp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp.label, random_state=42)
    print(f"[split] train={len(train)} val={len(val)} test={len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Evaluation helpers (shared by every model => fair comparison)
# ----------------------------------------------------------------------------
def compute_metrics(y_true, probs):
    """probs = P(fake) for each sample."""
    preds = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, preds), "precision": p, "recall": r,
            "f1": f1, "roc_auc": roc_auc_score(y_true, probs)}


def plot_confusion(y_true, probs, name, out_dir):
    cm = confusion_matrix(y_true, (probs >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(LABELS); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix: {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"cm_{name}.png"), dpi=120); plt.close(fig)


def plot_roc(all_probs, y_true, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, probs):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=120); plt.close(fig)


def plot_history(hist, name, out_dir):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(hist["train_loss"], label="train loss"); ax.plot(hist["val_f1"], label="val F1")
    ax.set_xlabel("epoch"); ax.set_title(f"Training curve: {name}"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"history_{name}.png"), dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------
# 3. MODEL 1: TF-IDF + Logistic Regression (baseline)
# ----------------------------------------------------------------------------
def run_baseline(train, val, test, out_dir):
    """Baseline = 'how far can simple word counts go?'. Needed to PROVE deep models help."""
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    Xtr = vec.fit_transform(train.content)                       # fit ONLY on train (no leakage)
    clf = LogisticRegression(max_iter=1000, C=3).fit(Xtr, train.label)
    probs = clf.predict_proba(vec.transform(test.content))[:, 1]

    # Explainability: the model is linear, so coefficients = word importance
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    print("[baseline] top words -> REAL:", list(names[order[:12]]))
    print("[baseline] top words -> FAKE:", list(names[order[-12:]]))
    return probs


# ----------------------------------------------------------------------------
# 4. MODEL 2: BiLSTM + Attention
# ----------------------------------------------------------------------------
class Vocab:
    """Word -> integer id. Built from TRAIN only. 0=<pad>, 1=<unk>."""
    def __init__(self, texts, max_size=30000, min_freq=2):
        cnt = Counter(w for t in texts for w in self.tok(t))
        words = [w for w, c in cnt.most_common(max_size) if c >= min_freq]
        self.itos = ["<pad>", "<unk>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @staticmethod
    def tok(t):
        return re.findall(r"[a-z0-9']+", t.lower())

    def encode(self, t, max_len):
        ids = [self.stoi.get(w, 1) for w in self.tok(t)][:max_len]
        return ids + [0] * (max_len - len(ids))


class BiLSTMAttention(nn.Module):
    """
    Embedding -> BiLSTM (reads text left->right AND right->left)
              -> Attention (learns which words matter) -> Dropout -> Linear -> 2 classes
    Attention weights double as an explanation of the prediction.
    """
    def __init__(self, vocab_size, emb=128, hid=128, layers=2, drop=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, layers, batch_first=True, bidirectional=True, dropout=drop)
        self.attn = nn.Linear(2 * hid, 1)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * hid, 2)

    def forward(self, x):
        mask = x != 0
        h, _ = self.lstm(self.drop(self.emb(x)))                         # (B, T, 2H)
        score = self.attn(h).squeeze(-1).masked_fill(~mask, -1e9)         # (B, T)
        w = torch.softmax(score, dim=1)                                   # attention weights
        ctx = (h * w.unsqueeze(-1)).sum(dim=1)                            # weighted sum
        return self.fc(self.drop(ctx)), w


@torch.no_grad()
def predict_lstm(model, loader, device):
    model.eval()
    out = []
    for (x,) in loader:
        logits, _ = model(x.to(device))
        out.append(torch.softmax(logits, -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def run_lstm(train, val, test, out_dir, device, epochs=6, max_len=300, bs=64):
    vocab = Vocab(train.content)
    enc = lambda df: torch.tensor([vocab.encode(t, max_len) for t in df.content])
    Xtr, Xva, Xte = enc(train), enc(val), enc(test)
    ytr = torch.tensor(train.label.values)
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva), batch_size=bs)
    te_loader = DataLoader(TensorDataset(Xte), batch_size=bs)

    model = BiLSTMAttention(len(vocab.itos)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state, bad, hist = 0, None, 0, {"train_loss": [], "val_f1": []}

    for ep in range(epochs):
        model.train(); total = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x)[0], y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)             # avoids exploding gradients
            opt.step(); total += loss.item()
        vf1 = compute_metrics(val.label.values, predict_lstm(model, va_loader, device))["f1"]
        hist["train_loss"].append(total / len(tr_loader)); hist["val_f1"].append(vf1)
        print(f"[lstm] epoch {ep+1}/{epochs} loss={hist['train_loss'][-1]:.4f} val_f1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:                                                  # early stopping
                print("[lstm] early stop"); break
    model.load_state_dict(best_state)
    plot_history(hist, "bilstm", out_dir)
    probs = predict_lstm(model, te_loader, device)

    # Explainability: which words did the attention focus on?
    print("\n[lstm] Attention explanations (top-8 words per article):")
    model.eval()
    for i in range(3):
        x = Xte[i:i + 1].to(device)
        with torch.no_grad():
            _, w = model(x)
        top = w[0].topk(8).indices.cpu().tolist()
        words = [vocab.itos[x[0, j].item()] for j in top]
        print(f"  true={LABELS[test.label[i]]} pred={LABELS[int(probs[i] >= .5)]} -> {words}")
    return probs


# ----------------------------------------------------------------------------
# 5. MODEL 3: BERT fine-tuning
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_bert(model, loader, device):
    model.eval()
    out = []
    for ids, mask in loader:
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        out.append(torch.softmax(logits, -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def run_bert(train, val, test, out_dir, device, name="bert-base-uncased",
             epochs=3, max_len=256, bs=16, lr=2e-5):
    """
    Fine-tuning = take a model pre-trained on huge text (it already 'knows English'),
    add a small classification layer on the [CLS] token, and train everything for a few
    epochs with a very small learning rate (2e-5) so we don't destroy the pre-trained knowledge.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)

    def encode(df):
        e = tok(list(df.content), truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt")
        return e["input_ids"], e["attention_mask"]

    print("[bert] tokenizing...", flush=True)
    tr_ids, tr_mask = encode(train); va_ids, va_mask = encode(val); te_ids, te_mask = encode(test)
    tr_loader = DataLoader(TensorDataset(tr_ids, tr_mask, torch.tensor(train.label.values)),
                           batch_size=bs, shuffle=True)
    va_loader = DataLoader(TensorDataset(va_ids, va_mask), batch_size=bs * 2)
    te_loader = DataLoader(TensorDataset(te_ids, te_mask), batch_size=bs * 2)

    print(f"[bert] training started on {device} ({len(tr_loader)} steps per epoch)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(tr_loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)  # warm-up then linear decay
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)                    # mixed precision = faster on GPU
    best_f1, best_state, hist = 0, None, {"train_loss": [], "val_f1": []}

    for ep in range(epochs):
        model.train(); total = 0; t0 = time.time()
        for step, (ids, mask, y) in enumerate(tr_loader, 1):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item()
            if step % 10 == 0 or step == len(tr_loader):
                el = time.time() - t0
                eta = el / step * (len(tr_loader) - step)
                print(f"[bert] epoch {ep+1}/{epochs} step {step}/{len(tr_loader)} "
                      f"loss={loss.item():.4f} | {el/60:.1f} min elapsed, ~{eta/60:.1f} min left", flush=True)
        print("[bert] validating...", flush=True)
        vf1 = compute_metrics(val.label.values, predict_bert(model, va_loader, device))["f1"]
        hist["train_loss"].append(total / len(tr_loader)); hist["val_f1"].append(vf1)
        print(f"[bert] epoch {ep+1}/{epochs} loss={hist['train_loss'][-1]:.4f} val_f1={vf1:.4f}")
        if vf1 > best_f1:                                                  # keep best epoch (by VALIDATION)
            best_f1 = vf1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    plot_history(hist, "bert", out_dir)
    return predict_bert(model, te_loader, device)


# ----------------------------------------------------------------------------
# 6. Main: run models, compare, error analysis
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./results")
    ap.add_argument("--models", default="baseline,lstm,bert")
    ap.add_argument("--bert_name", default="bert-base-uncased")
    ap.add_argument("--max_samples", type=int, default=None, help="use fewer rows for a quick test")
    ap.add_argument("--bert_epochs", type=int, default=3)
    ap.add_argument("--lstm_epochs", type=int, default=6)
    ap.add_argument("--bert_max_len", type=int, default=256, help="shorter = faster (try 128 on a Mac)")
    ap.add_argument("--bert_bs", type=int, default=16)
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda (use cpu if mps gives an error)")
    args = ap.parse_args()

    set_seed()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")                  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")                   # Apple Silicon Mac GPU
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    df = load_data(args.data_dir, args.max_samples)
    train, val, test = split_data(df)
    y = test.label.values

    wanted = args.models.split(",")
    probs = {}
    if "baseline" in wanted:
        probs["TFIDF+LogReg"] = run_baseline(train, val, test, args.out_dir)
    if "lstm" in wanted:
        probs["BiLSTM+Attn"] = run_lstm(train, val, test, args.out_dir, device, epochs=args.lstm_epochs)
    if "bert" in wanted:
        probs["BERT"] = run_bert(train, val, test, args.out_dir, device,
                                 name=args.bert_name, epochs=args.bert_epochs,
                                 max_len=args.bert_max_len, bs=args.bert_bs)

    # ---- comparison table ----
    rows = {n: compute_metrics(y, p) for n, p in probs.items()}
    table = pd.DataFrame(rows).T.round(4)
    print("\n===== TEST SET RESULTS =====\n", table)
    table.to_csv(os.path.join(args.out_dir, "results.csv"))
    for n, p in probs.items():
        plot_confusion(y, p, n.replace("+", "_"), args.out_dir)
    plot_roc(probs, y, args.out_dir)

    # ---- error analysis on the best model ----
    best = table["f1"].idxmax()
    wrong = test.assign(prob_fake=probs[best], pred=(probs[best] >= .5).astype(int))
    wrong = wrong[wrong.pred != wrong.label].copy()
    wrong["confidence"] = np.where(wrong.pred == 1, wrong.prob_fake, 1 - wrong.prob_fake)
    wrong = wrong.sort_values("confidence", ascending=False)
    wrong.to_csv(os.path.join(args.out_dir, f"errors_{best.replace('+', '_')}.csv"), index=False)
    print(f"\n[errors] best model = {best}; {len(wrong)} mistakes saved to results/")
    for _, r in wrong.head(3).iterrows():
        print(f"  actual={LABELS[r.label]} pred={LABELS[r.pred]} conf={r.confidence:.2f} | {r.content[:140]}...")


if __name__ == "__main__":
    main()
