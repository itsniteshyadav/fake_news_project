"""
Neural Fake News Detection Platform Server
Layman-friendly inference API supporting:
  1. DistilBERT Transformer (Advanced AI)
  2. BiLSTM with Self-Attention (Deep Learning Sequence Model)
  3. TF-IDF + Logistic Regression (Keyword Counting Model)
"""
import os
import json
import re
import torch
import joblib
from flask import Flask, request, jsonify, send_from_directory
from models_common import clean_text, Vocab, BiLSTMAttention, LABELS

app = Flask(__name__, static_folder="vercel_app/public", static_url_path="")

# Hardware acceleration setup
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[INIT] Compute Device: {device}")

# Global model references
tfidf_bundle = None
bilstm_model = None
bilstm_vocab = None
bert_model = None
bert_tokenizer = None

def load_models():
    global tfidf_bundle, bilstm_model, bilstm_vocab, bert_model, bert_tokenizer

    # 1. Load TF-IDF Baseline
    tfidf_path = "saved_models/tfidf/tfidf_model.joblib"
    if os.path.exists(tfidf_path):
        print("[LOAD] Initializing TF-IDF baseline...")
        tfidf_bundle = joblib.load(tfidf_path)

    # 2. Load BiLSTM + Attention
    bilstm_path = "saved_models/bilstm/bilstm_attention.pt"
    vocab_path = "saved_models/bilstm/vocab.json"
    if os.path.exists(bilstm_path) and os.path.exists(vocab_path):
        print("[LOAD] Initializing BiLSTM + Attention model...")
        bilstm_vocab = Vocab.load(vocab_path)
        bilstm_model = BiLSTMAttention(vocab_size=len(bilstm_vocab.itos), emb=128, hid=128, layers=2, drop=0.3)
        bilstm_model.load_state_dict(torch.load(bilstm_path, map_location="cpu"))
        bilstm_model.to(device)
        bilstm_model.eval()

    # 3. Load DistilBERT Transformer
    bert_dir = "saved_models/distilbert"
    if os.path.exists(bert_dir):
        print("[LOAD] Initializing DistilBERT transformer...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        bert_tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir, num_labels=2, output_attentions=True)
        bert_model.to(device)
        bert_model.eval()

    print("[SYSTEM] All models ready for inference.")


# ----------------------------------------------------------------------------
# Helper for classifying words into Fake-leaning vs Real-leaning
# ----------------------------------------------------------------------------
def categorize_words(words_list):
    """Categorizes a list of words using the TF-IDF learned dictionary if available."""
    if not tfidf_bundle:
        return words_list, []
    vec = tfidf_bundle["vectorizer"]
    clf = tfidf_bundle["classifier"]
    vocab = vec.vocabulary_
    coefs = clf.coef_[0]

    fake_words = []
    real_words = []
    for w in words_list:
        clean_w = w.lower().strip(",.!?\"'();:")
        if clean_w in vocab:
            score = coefs[vocab[clean_w]]
            if score > 0.05:
                fake_words.append(clean_w)
            elif score < -0.05:
                real_words.append(clean_w)
    return list(dict.fromkeys(fake_words)), list(dict.fromkeys(real_words))


# ----------------------------------------------------------------------------
# Inference Pipelines
# ----------------------------------------------------------------------------
def predict_tfidf(text):
    if not tfidf_bundle:
        return {"error": "Model weights unavailable."}
    vec = tfidf_bundle["vectorizer"]
    clf = tfidf_bundle["classifier"]

    cleaned = clean_text(text)
    x = vec.transform([cleaned])
    prob_fake = float(clf.predict_proba(x)[0, 1])
    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    conf = prob_fake if label == "FAKE" else 1.0 - prob_fake

    # Key words that pushed towards Fake vs Real
    names = vec.get_feature_names_out()
    c = x.multiply(clf.coef_[0]).tocoo()
    pairs = sorted(zip(c.col, c.data), key=lambda t: -abs(t[1]))[:12]

    fake_words = []
    real_words = []
    for col_idx, impact in pairs:
        word = names[col_idx]
        if impact > 0:
            fake_words.append(word)
        elif impact < 0:
            real_words.append(word)

    return {
        "model_name": "Keyword Counting Model",
        "label": label,
        "prob_fake": round(prob_fake, 4),
        "confidence": round(conf, 4),
        "words_making_fake": fake_words[:6],
        "words_making_real": real_words[:6]
    }


def predict_bilstm(text):
    if not bilstm_model or not bilstm_vocab:
        return {"error": "Model weights unavailable."}
    cleaned = clean_text(text)
    ids, tokens = bilstm_vocab.encode(cleaned, max_len=200)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, attn_weights = bilstm_model(x)
        prob_fake = float(torch.softmax(logits, dim=-1)[0, 1].item())

    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    conf = prob_fake if label == "FAKE" else 1.0 - prob_fake

    # Top words the attention layer focused on
    weights = attn_weights[0].cpu().numpy()
    valid_len = min(len(tokens), len(weights))
    sorted_idx = sorted(range(valid_len), key=lambda i: -weights[i])
    top_attended = [tokens[i] for i in sorted_idx if ids[i] > 1][:12]

    fake_words, real_words = categorize_words(top_attended)
    if not fake_words and label == "FAKE":
        fake_words = top_attended[:4]
    if not real_words and label == "REAL":
        real_words = top_attended[:4]

    return {
        "model_name": "Deep Learning Model (BiLSTM)",
        "label": label,
        "prob_fake": round(prob_fake, 4),
        "confidence": round(conf, 4),
        "words_making_fake": fake_words[:6],
        "words_making_real": real_words[:6]
    }


def predict_bert(text):
    if not bert_model or not bert_tokenizer:
        return {"error": "Model weights unavailable."}
    cleaned = clean_text(text)
    inputs = bert_tokenizer([cleaned], truncation=True, padding=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits
        prob_fake = float(torch.softmax(logits, dim=-1)[0, 1].item())

        # Extract [CLS] attention from the last layer to identify influential words
        last_attn = outputs.attentions[-1][0].mean(dim=0)
        cls_attn = last_attn[0].cpu().numpy()
        token_strs = bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    conf = prob_fake if label == "FAKE" else 1.0 - prob_fake

    # Clean tokens (remove subwords ##, [CLS], [SEP])
    token_pairs = sorted(zip(token_strs, cls_attn), key=lambda p: -p[1])
    filtered_tokens = []
    for t, w in token_pairs:
        if t not in ("[CLS]", "[SEP]", "[PAD]") and len(t) > 2 and not t.startswith("##"):
            filtered_tokens.append(t)

    fake_words, real_words = categorize_words(filtered_tokens[:12])
    if not fake_words and label == "FAKE":
        fake_words = filtered_tokens[:4]
    if not real_words and label == "REAL":
        real_words = filtered_tokens[:4]

    return {
        "model_name": "Advanced AI (DistilBERT)",
        "label": label,
        "prob_fake": round(prob_fake, 4),
        "confidence": round(conf, 4),
        "words_making_fake": fake_words[:6],
        "words_making_real": real_words[:6]
    }


# ----------------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("vercel_app/public", "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("vercel_app/public", filename)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    selected_model = data.get("model", "all").lower()

    if len(text.split()) < 5:
        return jsonify({"error": "Please enter at least 5 words of news text."}), 400

    results = {}

    if selected_model in ("bert", "distilbert", "all"):
        results["bert"] = predict_bert(text)

    if selected_model in ("bilstm", "all"):
        results["bilstm"] = predict_bilstm(text)

    if selected_model in ("tfidf", "all"):
        results["tfidf"] = predict_tfidf(text)

    return jsonify({
        "selected_model": selected_model,
        "results": results
    })

@app.route("/api/metrics", methods=["GET"])
def metrics():
    metrics_path = "saved_models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics file not found"}), 404

if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 3000))
    print(f"\n[SERVER ONLINE] Running on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
