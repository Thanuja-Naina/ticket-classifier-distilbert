"""Live demo of the fine-tuned model.  Run locally:  streamlit run demo/app.py

The model is NOT stored in GitHub (it is ~250 MB). It is loaded from the Hugging Face Hub,
where push_to_hub.py published it.
"""
import streamlit as st
from transformers import pipeline

MODEL_ID = "ThanuNaina/distilbert-banking77"    # <-- CHANGE THIS to the name you used in push_to_hub.py

st.title("Banking Support-Ticket Classifier")
st.write("Type a customer message. The model predicts which of 77 banking intents it belongs to.")


@st.cache_resource          # load the model once, not on every click
def load_model():
    return pipeline("text-classification", model=MODEL_ID, top_k=3)


classifier = load_model()

examples = ["I was charged twice for the same purchase",
            "My card has not arrived yet",
            "Why was my top up reversed?"]
chosen = st.selectbox("Try an example (or type your own below)", examples)
text = st.text_area("Customer message", chosen)

if st.button("Classify") and text.strip():
    for result in classifier(text)[0]:                       # top 3 predictions
        st.write(f"**{result['label']}**  -  confidence {result['score']:.1%}")
