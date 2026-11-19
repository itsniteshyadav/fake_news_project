"""
Interactive Fake News Detection CLI Demo
Allows selecting between:
  [1] Model 1: TF-IDF + Logistic Regression (Fast Baseline)
  [2] Model 2: BiLSTM + Attention (Sequential Deep Learning)
  [3] Model 3: DistilBERT (Transformer - State of the Art)
  [4] Run All 3 Models & Compare Side-by-Side
"""
import os
import sys
import torch
import joblib
from models_common import clean_text, Vocab, BiLSTMAttention

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# Model paths
TFIDF_PATH = "saved_models/tfidf/tfidf_model.joblib"
BILSTM_PATH = "saved_models/bilstm/bilstm_attention.pt"
VOCAB_PATH = "saved_models/bilstm/vocab.json"
BERT_PATH = "saved_models/distilbert"

print("\n" + "="*60)
print("  NEWSX - INTERACTIVE FAKE NEWS DETECTION TESTER")
print("  Tagline: See Beyond the Headlines.")
print("="*60)

# Check models
loaded_models = {}

if os.path.exists(TFIDF_PATH):
    print("Loading TF-IDF model...")
    loaded_models["tfidf"] = joblib.load(TFIDF_PATH)

if os.path.exists(BILSTM_PATH) and os.path.exists(VOCAB_PATH):
    print("Loading BiLSTM + Attention model...")
    vocab = Vocab.load(VOCAB_PATH)
    bilstm = BiLSTMAttention(vocab_size=len(vocab.itos), emb=128, hid=128, layers=2, drop=0.3)
    bilstm.load_state_dict(torch.load(BILSTM_PATH, map_location="cpu"))
    bilstm.to(device).eval()
    loaded_models["bilstm"] = {"model": bilstm, "vocab": vocab}

if os.path.exists(BERT_PATH):
    print("Loading DistilBERT model...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(BERT_PATH)
    bert = AutoModelForSequenceClassification.from_pretrained(BERT_PATH, num_labels=2)
    bert.to(device).eval()
    loaded_models["bert"] = {"model": bert, "tokenizer": tok}

print("All available models loaded successfully!\n")

def predict_tfidf(text):
    bundle = loaded_models["tfidf"]
    vec, clf = bundle["vectorizer"], bundle["classifier"]
    x = vec.transform([clean_text(text)])
    p_fake = float(clf.predict_proba(x)[0, 1])
    label = "FAKE" if p_fake >= 0.5 else "REAL"
    conf = p_fake if label == "FAKE" else 1.0 - p_fake

    # Key words
    names = vec.get_feature_names_out()
    c = x.multiply(clf.coef_[0]).tocoo()
    pairs = sorted(zip(c.col, c.data), key=lambda t: -abs(t[1]))[:6]
    fake_words = [names[i] for i, v in pairs if v > 0]
    real_words = [names[i] for i, v in pairs if v < 0]
    return label, conf, fake_words, real_words

def predict_bilstm(text):
    bundle = loaded_models["bilstm"]
    m, v = bundle["model"], bundle["vocab"]
    ids, tokens = v.encode(clean_text(text), max_len=200)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, w = m(x)
        p_fake = float(torch.softmax(logits, dim=-1)[0, 1].item())
    label = "FAKE" if p_fake >= 0.5 else "REAL"
    conf = p_fake if label == "FAKE" else 1.0 - p_fake

    top_idx = w[0].topk(min(6, len(tokens))).indices.cpu().tolist()
    top_words = [tokens[j] for j in top_idx if j < len(tokens) and ids[j] > 1]
    return label, conf, top_words

def predict_bert(text):
    bundle = loaded_models["bert"]
    m, tok = bundle["model"], bundle["tokenizer"]
    inputs = tok([clean_text(text)], truncation=True, padding=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = m(**inputs).logits
        p_fake = float(torch.softmax(logits, dim=-1)[0, 1].item())
    label = "FAKE" if p_fake >= 0.5 else "REAL"
    conf = p_fake if label == "FAKE" else 1.0 - p_fake
    return label, conf

def run_prediction(text, choice):
    print("\n" + "-"*60)
    if choice in ("1", "4") and "tfidf" in loaded_models:
        lbl, conf, fw, rw = predict_tfidf(text)
        print(f"[Model 1: TF-IDF]     Verdict: {lbl:<4} (Confidence: {conf*100:.1f}%)")
        if fw: print(f"                     Words pushing towards FAKE: {fw}")
        if rw: print(f"                     Words pushing towards REAL: {rw}")

    if choice in ("2", "4") and "bilstm" in loaded_models:
        lbl, conf, attn = predict_bilstm(text)
        print(f"[Model 2: BiLSTM]     Verdict: {lbl:<4} (Confidence: {conf*100:.1f}%)")
        if attn: print(f"                     Attention focused on words: {attn}")

    if choice in ("3", "4") and "bert" in loaded_models:
        lbl, conf = predict_bert(text)
        print(f"[Model 3: DistilBERT] Verdict: {lbl:<4} (Confidence: {conf*100:.1f}%)")
    print("-"*60 + "\n")

def main():
    while True:
        print("Choose an option:")
        print("  1 = TF-IDF + Logistic Regression")
        print("  2 = BiLSTM + Attention")
        print("  3 = DistilBERT Transformer")
        print("  4 = Compare All 3 Models Side-by-Side")
        print("  q = Quit")
        choice = input("Select (1-4, q) > ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        if choice not in ("1", "2", "3", "4"):
            print("Invalid choice. Try again.\n")
            continue

        text = input("\nEnter news text > ").strip()
        if len(text.split()) < 5:
            print("Please enter at least 5 words.\n")
            continue
        run_prediction(text, choice)

if __name__ == "__main__":
    main()
