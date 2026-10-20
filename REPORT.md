# Support-Ticket Classifier: Model Recommendation Report
_Fill in after your real runs. Delete the instructions in italics._

## 1. Business question
A bank's support team receives thousands of customer messages. Routing each message to the right team (card, transfers, top-ups...) automatically saves agent time. **Which model should the bank use?**

## 2. Results
| Model | Accuracy | Macro-F1 | Needs training? | Cost / speed |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | | | yes (seconds) | _very cheap, very fast_ |
| DistilBERT fine-tuned | | | yes (minutes on GPU) | _cheap to run, fast_ |
| LLM zero-shot (Groq) | | | no | _pay per request, slower_ |

## 3. Error analysis
_Which intent pairs are confused most (from `results/distilbert_top_confusions.csv`)? Give 2 examples and a reason._

## 4. Recommendation
_Pick one model and defend it using YOUR numbers. Example: "Use fine-tuned DistilBERT: it is X points more accurate than the baseline, runs on a CPU, and has no per-request cost, unlike the LLM which scored Y on a 200-question sample." Mention one next step (more data for weak intents, human review for low-confidence predictions)._

## 5. Limits
_The LLM was tested on 200 questions only; Banking77 is English-only and one domain; no live traffic was tested._
