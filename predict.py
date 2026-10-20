"""Try the fine-tuned model.  Run: python predict.py "I was charged twice for the same purchase" """
import sys

from transformers import pipeline

classifier = pipeline("text-classification", model="outputs/distilbert-banking77", top_k=3)
for text in sys.argv[1:] or ["My card has not arrived yet"]:
    print(text, "->", [(r["label"], round(r["score"], 3)) for r in classifier(text)[0]])
