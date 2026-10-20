"""Model 2: fine-tune DistilBERT.  Run: python finetune.py

Fine-tuning = take a model that already understands English (distilbert-base-uncased)
and train it a little more on OUR task (77 banking intents).
Use a GPU (free Google Colab T4 takes ~10 minutes). On a normal laptop CPU it is very slow.
"""
import json
import os

import numpy as np
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments)

from utils import load_data, score, top_confusions

# make sure the output folders exist (they are missing if you uploaded the files by hand)
os.makedirs("results", exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 5

train, validation, test, label_names = load_data()

# 1) Tokenizer: turns text into the ids the model understands
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def to_dataset(df):
    dataset = Dataset.from_pandas(df)
    return dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=64), batched=True)


train_ds, val_ds, test_ds = to_dataset(train), to_dataset(validation), to_dataset(test)

# 2) Model: DistilBERT with a new output layer for our 77 labels
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_names),
    id2label=dict(enumerate(label_names)), label2id={name: i for i, name in enumerate(label_names)})


def compute_metrics(prediction_output):
    predictions = np.argmax(prediction_output.predictions, axis=-1)
    result = score(prediction_output.label_ids, predictions)
    return {"accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}


# 3) Training settings
args = TrainingArguments(
    output_dir="outputs/distilbert-banking77",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    eval_strategy="epoch",              # check the validation set after every epoch
    save_strategy="epoch",
    load_best_model_at_end=True,        # keep the best epoch, not just the last one
    metric_for_best_model="macro_f1",
    save_total_limit=1,
    report_to="none",
    seed=42,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                  data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
trainer.train()

# 4) Final evaluation on the test set (data the model has never seen)
predictions = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
metrics = score(test["label"], predictions)
print("Fine-tuned DistilBERT:", metrics)
json.dump(metrics, open("results/distilbert_metrics.json", "w"), indent=2)
top_confusions(test["label"], predictions, label_names).to_csv("results/distilbert_top_confusions.csv", index=False)

trainer.save_model("outputs/distilbert-banking77")
tokenizer.save_pretrained("outputs/distilbert-banking77")
print("Model saved. To publish it, run: python push_to_hub.py your-username/distilbert-banking77")
