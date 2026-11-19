# Fake News Detection System: TF-IDF vs. BiLSTM+Attention vs. BERT

An end-to-end Machine Learning and Deep Learning system to detect whether a news article is fake or genuine, featuring explainable AI mechanisms and an interactive multi-model web platform.

---

## 👥 Authors & Contact
* **Created By**: Nitesh Yadav & Abhinav Agarwal
* **Inquiries & Support**: `itsniteshyadav.dev@gmail.com`

---

## 🌟 Key Features

1. **Multi-Model Architecture**:
   * **Model 1: TF-IDF + Logistic Regression (Baseline)**: High-speed linear n-gram model that extracts directional word coefficients.
   * **Model 2: BiLSTM + Self-Attention (Sequential Deep Learning)**: Bidirectional LSTM that models sentence structure and outputs word-level attention weights for explainability.
   * **Model 3: DistilBERT Transformer (State of the Art)**: Pre-trained contextual language representation fine-tuned for classification with >99% test accuracy.

2. **Unified Interactive Web Platform**:
   * Dropdown selector allowing users to choose any individual model or run **"Compare All Models Side-by-Side"**.
   * One-click demo buttons with sample fake and real news articles.
   * Visual word breakdown drawers showing the exact words that influenced each model's prediction.

3. **Interactive Terminal CLI**:
   * Command-line testing utility (`python demo.py`) for quick terminal evaluations.

---

## 📊 Dataset: ISOT Fake News Dataset

* **Source**: University of Victoria's Information Security and Object Technology (ISOT) Research Lab.
* **Volume**: ~44,898 articles (`Fake.csv` and `True.csv`).
* **Preprocessing & Data Leakage Prevention**:
  * Combined `title` and `text`.
  * Stripped Reuters source markers (e.g., `WASHINGTON (Reuters) -`) to prevent models from taking shortcuts on publication tags.
  * Stratified split into **70% Train**, **15% Validation**, and **15% Test**.

---

## 🚀 Quickstart: How to Run the Website

### Prerequisites
* Python 3.9+ (or virtual environment with PyTorch and Transformers)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:3000
```

---

## 💻 Terminal CLI Usage

You can also run predictions directly from the command line:
```bash
python demo.py
```
Follow the interactive prompt to choose a model (`1`, `2`, `3`, or `4` to compare all).

---

## 📈 Benchmark Results (Test Set)

| Model Architecture | Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** | Transformer (SOTA) | **99.9%** | **0.998** | **1.000** | **0.999** | **0.999** |
| **TF-IDF + LogReg** | N-Gram Baseline | 98.3% | 0.990 | 0.972 | 0.981 | 0.998 |
| **BiLSTM + Attention** | Recurrent Neural Net | 97.2% | 0.975 | 0.963 | 0.969 | 0.995 |

---

## 📂 Project Structure

```
fake_news_project/
├── app.py                      # Flask web server supporting all 3 model options
├── demo.py                     # Interactive CLI tester
├── models_common.py            # Shared Vocab, BiLSTMAttention, and text cleaner
├── train_and_save_models.py    # Training & model export pipeline
├── requirements.txt            # Python dependencies
├── data/                       # Dataset directory (Fake.csv, True.csv)
├── saved_models/               # Saved model checkpoints
│   ├── tfidf/                  # Serialized TF-IDF vectorizer and LogReg classifier
│   ├── bilstm/                 # PyTorch BiLSTM weights + vocab.json
│   └── distilbert/             # HuggingFace Transformer model & tokenizer
├── vercel_app/
│   └── public/
│       └── index.html          # Web UI with model selector & visual results
└── results/                    # Research evaluation plots & confusion matrices
```
