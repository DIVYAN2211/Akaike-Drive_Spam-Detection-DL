
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.preprocessing import clean_text
import pickle
import os

MAX_LEN = 100
VOCAB_SIZE = 5000

def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1','v2']]
    df.columns = ['label','text']
    df['text'] = df['text'].apply(clean_text)
    df['label'] = df['label'].map({'ham':0,'spam':1})

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post')

    with open("tokenizer.pkl","wb") as f:
        pickle.dump(tokenizer,f)

    return X_train_seq, X_test_seq, y_train.values, y_test.values, VOCAB_SIZE
