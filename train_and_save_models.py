"""
Train & Export Script:
Trains TF-IDF + LogReg, BiLSTM + Attention, and DistilBERT.
Saves all model checkpoints into `saved_models/` so they can be loaded instantly in the web app and CLI.
"""
import os
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import joblib

from models_common import clean_text, Vocab, BiLSTMAttention, LABELS

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(data_dir="./data", max_samples=20000):
    print(f"[1/4] Loading dataset from {data_dir}...")
    fake = pd.read_csv(os.path.join(data_dir, "Fake.csv"))
    true = pd.read_csv(os.path.join(data_dir, "True.csv"))
    fake["label"], true["label"] = 1, 0
    df = pd.concat([fake, true], ignore_index=True)

    df["content"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).map(clean_text)
    df = df[df["content"].str.split().str.len() > 20]
    df = df.drop_duplicates(subset="content")
    df = df[["content", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)
    if max_samples:
        df = df.iloc[:max_samples]
    print(f"      Total samples: {len(df)} | Fake: {df.label.sum()} | Real: {(df.label == 0).sum()}")
    return df

def split_data(df):
    train, temp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp.label, random_state=42)
    print(f"      Splits: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def evaluate_predictions(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4)
    }

# ----------------------------------------------------------------------------
# 1. Model 1: TF-IDF + Logistic Regression
# ----------------------------------------------------------------------------
def train_and_save_tfidf(train, test, out_dir):
    print("\n[2/4] Training Model 1: TF-IDF + Logistic Regression...")
    os.makedirs(out_dir, exist_ok=True)
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    X_tr = vec.fit_transform(train.content)
    clf = LogisticRegression(max_iter=1000, C=3)
    clf.fit(X_tr, train.label)

    probs = clf.predict_proba(vec.transform(test.content))[:, 1]
    metrics = evaluate_predictions(test.label.values, probs)
    print(f"      TF-IDF Test Metrics: {metrics}")

    joblib.dump({"vectorizer": vec, "classifier": clf}, os.path.join(out_dir, "tfidf_model.joblib"))
    print(f"      Saved to {out_dir}/tfidf_model.joblib")
    return metrics

# ----------------------------------------------------------------------------
# 2. Model 2: BiLSTM + Attention
# ----------------------------------------------------------------------------
def train_and_save_bilstm(train, val, test, out_dir, device, epochs=4, max_len=200, bs=64):
    print(f"\n[3/4] Training Model 2: BiLSTM + Attention on {device}...")
    os.makedirs(out_dir, exist_ok=True)

    vocab = Vocab(train.content, max_size=25000)
    vocab.save(os.path.join(out_dir, "vocab.json"))

    def encode_df(df):
        return torch.tensor([vocab.encode(t, max_len)[0] for t in df.content])

    X_tr, X_va, X_te = encode_df(train), encode_df(val), encode_df(test)
    y_tr = torch.tensor(train.label.values)
    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va), batch_size=bs)
    te_loader = DataLoader(TensorDataset(X_te), batch_size=bs)

    model = BiLSTMAttention(vocab_size=len(vocab.itos), emb=128, hid=128, layers=2, drop=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        va_probs = []
        with torch.no_grad():
            for (x,) in va_loader:
                logits, _ = model(x.to(device))
                va_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        val_p = torch.cat(va_probs).numpy()
        val_m = evaluate_predictions(val.label.values, val_p)
        print(f"      Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(tr_loader):.4f} | Val F1: {val_m['f1']:.4f}")

        if val_m["f1"] >= best_f1:
            best_f1 = val_m["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(out_dir, "bilstm_attention.pt"))
    print(f"      Saved model checkpoint to {out_dir}/bilstm_attention.pt")

    # Final test evaluation
    model.to(device).eval()
    te_probs = []
    with torch.no_grad():
        for (x,) in te_loader:
            logits, _ = model(x.to(device))
            te_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    test_metrics = evaluate_predictions(test.label.values, torch.cat(te_probs).numpy())
    print(f"      BiLSTM Test Metrics: {test_metrics}")
    return test_metrics

# ----------------------------------------------------------------------------
# 3. Model 3: DistilBERT Fine-Tuning
# ----------------------------------------------------------------------------
def train_and_save_distilbert(train, val, test, out_dir, device, epochs=2, max_len=128, bs=32, lr=2e-5):
    print(f"\n[4/4] Fine-tuning Model 3: DistilBERT on {device}...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)

    def encode_texts(texts):
        enc = tokenizer(list(texts), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    tr_ids, tr_mask = encode_texts(train.content)
    va_ids, va_mask = encode_texts(val.content)
    te_ids, te_mask = encode_texts(test.content)

    tr_loader = DataLoader(TensorDataset(tr_ids, tr_mask, torch.tensor(train.label.values)), batch_size=bs, shuffle=True)
    va_loader = DataLoader(TensorDataset(va_ids, va_mask), batch_size=bs * 2)
    te_loader = DataLoader(TensorDataset(te_ids, te_mask), batch_size=bs * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(tr_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        t0 = time.time()
        for step, (ids, mask, y) in enumerate(tr_loader, 1):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask, labels=y)
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if step % 50 == 0 or step == len(tr_loader):
                elapsed = time.time() - t0
                print(f"      Epoch {epoch+1}/{epochs} | Step {step}/{len(tr_loader)} | Loss: {loss.item():.4f} ({elapsed:.1f}s)", flush=True)

    print("      Saving DistilBERT model and tokenizer...")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"      Saved DistilBERT to {out_dir}")

    # Evaluate on test set
    model.eval()
    te_probs = []
    with torch.no_grad():
        for ids, mask in te_loader:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            te_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    test_metrics = evaluate_predictions(test.label.values, torch.cat(te_probs).numpy())
    print(f"      DistilBERT Test Metrics: {test_metrics}")
    return test_metrics


def main():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Running model training on device: {device}")

    # Train on 16,000 samples for fast, high-accuracy training (~3 mins total)
    df = load_data(data_dir="./data", max_samples=16000)
    train, val, test = split_data(df)

    all_metrics = {}
    all_metrics["TFIDF_LogReg"] = train_and_save_tfidf(train, test, "saved_models/tfidf")
    all_metrics["BiLSTM_Attn"] = train_and_save_bilstm(train, val, test, "saved_models/bilstm", device, epochs=4)
    all_metrics["DistilBERT"] = train_and_save_distilbert(train, val, test, "saved_models/distilbert", device, epochs=2)

    with open("saved_models/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "="*50)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("Summary of Test Metrics:")
    print(pd.DataFrame(all_metrics).T)
    print("="*50)

if __name__ == "__main__":
    main()
