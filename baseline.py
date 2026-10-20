"""Model 1: TF-IDF + Logistic Regression (a classic, fast baseline).  Run: python baseline.py

TF-IDF turns each sentence into numbers based on how important its words are.
Logistic Regression then learns which words point to which intent.
"""
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from utils import load_data, score, top_confusions


def train_baseline(train_texts, train_labels):
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),   # single words and word pairs
                          LogisticRegression(max_iter=2000, C=10))
    return model.fit(train_texts, train_labels)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)   # create the folder if it is missing
    train, validation, test, label_names = load_data()
    model = train_baseline(train["text"], train["label"])
    predictions = model.predict(test["text"])

    metrics = score(test["label"], predictions)
    print("TF-IDF + Logistic Regression:", metrics)
    json.dump(metrics, open("results/baseline_metrics.json", "w"), indent=2)
    top_confusions(test["label"], predictions, label_names).to_csv("results/baseline_top_confusions.csv", index=False)
