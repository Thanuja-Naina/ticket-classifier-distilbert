"""Small helpers shared by all scripts: load the data and measure results."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def _download_banking77():
    """Download Banking77. Newer `datasets` versions (4.x) cannot run dataset scripts,
    so we load a parquet copy of the dataset instead (with a backup source)."""
    from datasets import load_dataset

    attempts = [
        {"path": "mteb/banking77"},                                         # public parquet copy (works with new `datasets`)
        {"path": "PolyAI/banking77", "revision": "refs/convert/parquet"},   # backup: converted version of the original
    ]
    for kwargs in attempts:
        try:
            return load_dataset(**kwargs)
        except Exception as error:
            print(f"Could not load {kwargs['path']}: {error}")
    raise RuntimeError("Could not download Banking77. Check your internet connection and try again.")


def _get_label_names(data):
    """The list of 77 intent names, in label-number order."""
    feature = data["train"].features["label"]
    if hasattr(feature, "names"):                       # normal case: labels know their names
        return list(feature.names)
    pairs = data["train"].to_pandas()[["label", "label_text"]].drop_duplicates().sort_values("label")
    return pairs["label_text"].tolist()                 # fallback: build names from the label_text column


def load_data():
    """Returns train, validation and test tables plus the list of label names."""
    data = _download_banking77()
    label_names = _get_label_names(data)
    train = data["train"].to_pandas()[["text", "label"]]
    test = data["test"].to_pandas()[["text", "label"]]
    # keep 10% of train for validation (used to pick the best epoch)
    train, validation = train_test_split(train, test_size=0.1, stratify=train["label"], random_state=42)
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True), label_names


def score(y_true, y_pred):
    """Accuracy = share of correct answers. Macro-F1 = average F1 over all 77 classes (treats each class equally)."""
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "n_test_examples": len(y_true)}


def top_confusions(y_true, y_pred, label_names, n=10):
    """Which pairs of intents does the model mix up most often?"""
    table = pd.DataFrame({"true": [label_names[i] for i in y_true], "predicted": [label_names[i] for i in y_pred]})
    wrong = table[table["true"] != table["predicted"]]
    return wrong.value_counts().head(n).reset_index(name="count")
