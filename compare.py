"""Put the three models side by side.  Run: python compare.py   (after running the three model scripts)"""
import json
import os

files = {"TF-IDF + Logistic Regression": "results/baseline_metrics.json",
         "DistilBERT (fine-tuned)": "results/distilbert_metrics.json",
         "LLM zero-shot (Groq)": "results/llm_zero_shot_metrics.json"}

os.makedirs("results", exist_ok=True)
lines = ["| Model | Accuracy | Macro-F1 | Test examples |", "|---|---|---|---|"]
for name, path in files.items():
    if os.path.exists(path):
        m = json.load(open(path))
        lines.append(f"| {name} | {m['accuracy']} | {m['macro_f1']} | {m['n_test_examples']} |")

table = "\n".join(lines)
print(table)
open("results/model_comparison.md", "w").write(table + "\n")
