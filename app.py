import streamlit as st
import torch
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.spam_detector import SpamDetector
from utils.preprocessing import clean_text

# Must match training config
MAX_LEN = 100
VOCAB_SIZE = 5000  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
   
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

   
    model = SpamDetector(VOCAB_SIZE).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()

    return model, tokenizer



st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
st.title("📩 SMS Spam Detection System")
st.write("Enter a message below to check whether it is Spam or Ham.")

message = st.text_area("Enter SMS Message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        model, tokenizer = load_model()

        # Clean text
        cleaned = clean_text(message)

        # Convert to sequence
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        tensor = torch.tensor(padded).long().to(device)

        # Predict
        with torch.no_grad():
            output = model(tensor).squeeze()
            prob = torch.sigmoid(output).item()

        # Display result
        if prob > 0.3:
            st.error(f"🚨 Spam Detected")
            st.write(f"Confidence: {prob:.2f}")
        else:
            st.success(f"✅ Legitimate (Ham) Message")
            st.write(f"Confidence: {1 - prob:.2f}")
