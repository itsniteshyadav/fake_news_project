"""Run once on your Mac (from the project folder with ./data): exports the trained TF-IDF + LogReg model to lib/model.json."""
import json, os, sys
sys.path.insert(0, "..")  # reuse the project's cleaning/split code if run from vercel_app/
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from fake_news_detection import load_data, split_data, compute_metrics

data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"
train, val, test = split_data(load_data(data_dir))
vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
clf = LogisticRegression(max_iter=1000, C=3).fit(vec.fit_transform(train.content), train.label)
probs = clf.predict_proba(vec.transform(test.content))[:, 1]
m = {k: round(float(v), 4) for k, v in compute_metrics(test.label.values, probs).items()}
model = {"terms": {t: [round(float(i), 6), round(float(c), 6)]
                   for t, i, c in zip(vec.get_feature_names_out(), vec.idf_, clf.coef_[0])},
         "intercept": float(clf.intercept_[0]), "stop": sorted(ENGLISH_STOP_WORDS), "metrics": m}
os.makedirs("lib", exist_ok=True)
json.dump(model, open("lib/model.json", "w"))
json.dump([{"text": t, "p": float(p)} for t, p in zip(test.content[:5], probs[:5])], open("lib/parity.json", "w"))
print("Saved lib/model.json | test metrics:", m)
