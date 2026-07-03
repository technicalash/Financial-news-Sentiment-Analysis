import streamlit as st
import re
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

def clean_data(text):
  text=re.sub(r"/r/n"," ",text)
  text=re.sub(r"http\S+"," ",text)
  text=re.sub(r"<.*?>"," ",text)
  text=re.sub(r"/s+"," ",text)
  return text

id2label = {
    0:"Negative",
    1:"Neutral",
    2:"Positive"
}

tokenizer = AutoTokenizer.from_pretrained("finbert_sentiment_model")
model = AutoModelForSequenceClassification.from_pretrained(
    "finbert_sentiment_model"
)

model.eval()

st.set_page_config(
    page_title="Financial News Sentiment Analysis",
    layout="centered"
)

st.title("Financial News Sentiment Analysis")
st.write(
    "Check wheather a Financial news is positive, neutral or negative"
)
message = st.text_area(
    "Enter the news",
    height=100
)

if st.button("Check"):
    if message.strip() == "":
        st.warning("Please enter the news.")
    else:
        msg=clean_data(message)
        inputs = tokenizer(
            msg,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
            )
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        result = id2label[prediction]
        st.success(f"Predicted Sentiment: {result}")