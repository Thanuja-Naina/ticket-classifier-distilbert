# Support-Ticket Intent Classifier (DistilBERT)

Classifies customer-service questions into 77 banking intents (the **Banking77** dataset), for example "card not arrived" or "top up failed". Three approaches are compared on the same test set:

1. **TF-IDF + Logistic Regression**: a fast classic baseline
2. **Fine-tuned DistilBERT**: Hugging Face Transformers + PyTorch
3. **Zero-shot LLM**: Llama through the Groq API, with no training

**Stack:** Python, PyTorch, Hugging Face Transformers/Datasets, scikit-learn, Groq API, pytest, GitHub Actions

## Files

| File | What it does |
|---|---|
| `utils.py` | Loads the data, computes accuracy / macro-F1, finds the most confused intent pairs |
| `baseline.py` | Model 1: TF-IDF + Logistic Regression |
| `finetune.py` | Model 2: fine-tune DistilBERT (use a GPU) |
| `llm_zero_shot.py` | Model 3: Groq LLM on a 200-question sample |
| `compare.py` | Builds the results table for the three models |
| `demo/app.py`, `demo/requirements.txt` | Streamlit demo and its light dependency list |
| `push_to_hub.py` | Publishes the model to the Hugging Face Hub (public link)|
| `predict.py` | Try the fine-tuned model on your own sentence |
| `REPORT.md` | Model recommendation report for a business reader (fill in after your run) |

## Dataset note
The data is the public `mteb/banking77` copy of Banking77 (77 intents). It has 9,993 training and 3,076 test examples, slightly fewer than the original release (10,003 / 3,080). Quote these numbers if you report results.

## Run it
```bash
pip install -r requirements.txt
python baseline.py                              # about a minute
python finetune.py                              # use Google Colab (free GPU, ~10 min); CPU is very slow
export GROQ_API_KEY="your_key"                  # see .env.example; never commit your key
python llm_zero_shot.py
python compare.py                               # results table for all three models
python predict.py "I was charged twice for the same purchase"
```

**Published model:** _after training, publish it to the Hugging Face Hub (see DEPLOY.md) and paste the link here._

## Live demo
`demo/app.py` is a small Streamlit app that loads the published model from the Hugging Face Hub.
**Live demo:** _paste your streamlit.app link here after deploying_ | **Model:** _paste your Hugging Face link here_

## Business problem
Routing customer messages to the right team by hand is slow. An automatic classifier can route them instantly. The business question is **which model to use**: a cheap baseline, a fine-tuned model, or a general LLM. `REPORT.md` answers it with evidence.

## Results (fill in from the `results/` folder)

| Model | Accuracy | Macro-F1 | Test examples |
|---|---|---|---|
| TF-IDF + Logistic Regression | | | |
| DistilBERT (fine-tuned) | | | |
| LLM zero-shot (Groq) | | | 200 |

**Error analysis:** _open `results/distilbert_top_confusions.csv`, name the intent pairs the model mixes up most, and explain why (for example, two intents that differ by one detail)._

**Fair-comparison notes:** the LLM is tested on 200 questions, not the full test set, so its number is less certain. Answers that cannot be matched to a label count as wrong.

## Tests
`pytest -q` runs without a GPU or internet. It checks that the baseline learns a tiny dataset, that scoring is correct, and that LLM replies are mapped to labels correctly.
