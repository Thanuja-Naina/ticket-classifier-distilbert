import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("groq", types.SimpleNamespace(Groq=object))   # llm_zero_shot imports groq; not needed for these tests

from baseline import train_baseline
from llm_zero_shot import text_to_label_id
from utils import _get_label_names, score, top_confusions

NAMES = ["card_arrival", "lost_card", "exchange_rate"]


def test_baseline_learns_a_tiny_dataset():
    texts = ["where is my new card", "my card has not arrived", "i lost my card", "my card was stolen",
             "what exchange rate do you use", "how is the currency rate set"] * 5
    labels = [0, 0, 1, 1, 2, 2] * 5
    model = train_baseline(texts, labels)
    assert list(model.predict(["my card never arrived", "i lost my wallet and card"])) == [0, 1]


def test_score_and_confusions():
    y_true, y_pred = [0, 0, 1, 2], [0, 1, 1, 2]
    assert score(y_true, y_pred)["accuracy"] == 0.75
    confusions = top_confusions(y_true, y_pred, NAMES)
    assert confusions.iloc[0]["true"] == "card_arrival" and confusions.iloc[0]["predicted"] == "lost_card"


def test_llm_reply_is_mapped_to_a_label():
    assert text_to_label_id("card_arrival", NAMES) == 0
    assert text_to_label_id("The intent is exchange_rate.", NAMES) == 2
    assert text_to_label_id("lost_crd", NAMES) == 1          # small typo
    assert text_to_label_id("something unrelated", NAMES) is None


def test_label_names_from_classlabel_or_label_text_column():
    class FakeTrain:
        def __init__(self, features, frame=None):
            self.features, self._frame = features, frame
        def to_pandas(self):
            return self._frame

    with_names = {"train": FakeTrain({"label": types.SimpleNamespace(names=["a", "b"])})}
    assert _get_label_names(with_names) == ["a", "b"]

    import pandas as pd
    frame = pd.DataFrame({"label": [1, 0, 1], "label_text": ["lost_card", "card_arrival", "lost_card"]})
    without_names = {"train": FakeTrain({"label": object()}, frame)}
    assert _get_label_names(without_names) == ["card_arrival", "lost_card"]
