"""Publish your fine-tuned model to the Hugging Face Hub (a public link recruiters can open).

Run once, after finetune.py:   python push_to_hub.py your-username/distilbert-banking77
You need a free Hugging Face account and an access token with WRITE permission (Settings -> Access Tokens).
"""
import sys

from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer

repo_name = sys.argv[1]
login()   # paste your token when asked; never write it in the code

model = AutoModelForSequenceClassification.from_pretrained("outputs/distilbert-banking77")
tokenizer = AutoTokenizer.from_pretrained("outputs/distilbert-banking77")
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
print(f"Published: https://huggingface.co/{repo_name}")
