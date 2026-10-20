"""Model 3: ask a big LLM (Groq / Llama) to classify with NO training ("zero-shot").
Run: export GROQ_API_KEY="your_key"  then  python llm_zero_shot.py

We test on 200 random test questions (not all ~3000) to keep API usage small.
"""
import difflib
import json
import os
import time

from groq import Groq

from utils import load_data, score

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
SAMPLE_SIZE = 200


def text_to_label_id(reply, label_names):
    """Turn the LLM's text reply into a label number. Returns None if we cannot match it."""
    reply = reply.strip().lower().replace(" ", "_")
    names = [n.lower() for n in label_names]
    if reply in names:
        return names.index(reply)
    for i, name in enumerate(names):          # label name appears inside a longer reply
        if name in reply:
            return i
    close = difflib.get_close_matches(reply, names, n=1, cutoff=0.8)   # tolerate small typos
    return names.index(close[0]) if close else None


def classify(client, question, label_names):
    prompt = ("Classify this banking customer question into exactly ONE of these intents:\n"
              + ", ".join(label_names) + f"\n\nQuestion: {question}\n\nAnswer with the intent name only.")
    response = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=20,
                                              messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY first (see .env.example)")
    os.makedirs("results", exist_ok=True)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    _, _, test, label_names = load_data()
    sample = test.sample(SAMPLE_SIZE, random_state=42)

    true_labels, predicted_labels, unmatched = [], [], 0
    for question, label in zip(sample["text"], sample["label"]):
        try:
            predicted = text_to_label_id(classify(client, question, label_names), label_names)
        except Exception as error:            # e.g. rate limit: wait and count as wrong
            print("API error:", error)
            time.sleep(5)
            predicted = None
        if predicted is None:
            unmatched += 1
            predicted = len(label_names)      # an impossible label, so it counts as wrong
        true_labels.append(label)
        predicted_labels.append(predicted)
        time.sleep(0.2)                       # stay under the rate limit

    metrics = score(true_labels, predicted_labels)
    metrics["unmatched_answers"] = unmatched
    print("LLM zero-shot:", metrics)
    json.dump(metrics, open("results/llm_zero_shot_metrics.json", "w"), indent=2)
