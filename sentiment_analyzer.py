import json
import re
import string
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import os

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model.h5')
ARTIFACTS_PATH = os.path.join(BASE_DIR, 'preproc_artifacts.json')

try:
    stop_words_set = set(stopwords.words('english'))
    for w in ('no', 'not', 'nor'):
        stop_words_set.discard(w)
except LookupError:
    st.error("NLTK stopwords corpus not found. Please download it by running: import nltk; nltk.download('stopwords')")
    stop_words_set = set() 

snowball_stemmer = SnowballStemmer('english')

# --- Load Model and Artifacts (Cached) ---
@st.cache_resource
def load_sentiment_model_and_artifacts():
    """Loads the TensorFlow model and preprocessing artifacts."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Sentiment model file not found at {MODEL_PATH}")
        return None, None, None, None, None
    if not os.path.exists(ARTIFACTS_PATH):
        st.error(f"Preprocessing artifacts file not found at {ARTIFACTS_PATH}")
        return None, None, None, None, None
        
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with open(ARTIFACTS_PATH, 'r', encoding='utf-8') as f:
            artifacts = json.load(f)
        vocab = artifacts['vocab']
        max_len = artifacts['max_len']
        pad_idx = artifacts['pad_idx']
        unk_idx = artifacts['unk_idx']
        return model, vocab, max_len, pad_idx, unk_idx
    except Exception as e:
        st.error(f"Error loading sentiment model or artifacts: {e}")
        return None, None, None, None, None

# --- Preprocessing Functions (from user's script) ---
def datapreprocess(sen: str) -> str:
    sen = str(sen)
    # 1. Expand common contractions
    sen = re.sub(r"didn't", "did not", sen)
    sen = re.sub(r"don't", "do not", sen)
    sen = re.sub(r"won't", "will not", sen)
    sen = re.sub(r"can't", "can not", sen)
    sen = re.sub(r"wasn't", "was not", sen)
    sen = re.sub(r"\'ve", " have", sen)
    sen = re.sub(r"\'m", " am", sen)
    sen = re.sub(r"\'ll", " will", sen)
    sen = re.sub(r"\'re", " are", sen)
    sen = re.sub(r"\'s", " is", sen)
    sen = re.sub(r"\'d", " would", sen)
    sen = re.sub(r"\'t", " not", sen)
    
    # 2. Build a set of all punctuation + digits to remove
    punc_and_digits = set(string.punctuation)
    punc_and_digits.update(str(d) for d in range(10))
    
    # 3. Lowercase & split
    sen = sen.lower()
    words = sen.split()
    
    # 4. Remove punctuation/digits and non‑ascii
    cleaned = []
    for w in words:
        try:
            t = ''.join(ch for ch in w.encode('ascii', 'ignore').decode('ascii') 
                        if ch not in punc_and_digits)
            if t:  
                cleaned.append(t)
        except Exception:
            pass
    return " ".join(cleaned)

def full_preprocess_text(text: str) -> str:
    """Applies full preprocessing: clean, stopword removal, stemming."""
    if text is None or pd.isna(text):
        return ""
    text_str = str(text)
    
    clean = datapreprocess(text_str)
    tokens = [tok for tok in clean.split() if tok not in stop_words_set]
    stems = [snowball_stemmer.stem(tok) for tok in tokens]
    return " ".join(stems).strip()

def encode_texts_for_sentiment(texts, vocab_map, max_len, pad_idx, unk_idx):
    """Encodes preprocessed texts into sequences for the model."""
    sequences = []
    for txt in texts:
        seq = [vocab_map.get(tok, unk_idx) for tok in txt.split()]
        sequences.append(seq)
    return pad_sequences(
        sequences,
        maxlen=max_len,
        padding='post',
        truncating='post',
        value=pad_idx
    )

# --- Main Sentiment Prediction Function ---
def predict_sentiments_for_texts(texts_list: list[str]) -> list[str]:
    """
    Predicts sentiment for a list of text strings.
    Returns a list of "Positive", "Negative", or "N/A".
    """
    if not texts_list:
        return []

    model, vocab, max_len, pad_idx, unk_idx = load_sentiment_model_and_artifacts()
    if model is None:
        return ["Error: Model not loaded"] * len(texts_list)

    processed_texts = [full_preprocess_text(s) for s in texts_list]
    
    valid_texts_with_indices = []
    for i, s_processed in enumerate(processed_texts):
        if s_processed and s_processed.strip():
            valid_texts_with_indices.append((i, s_processed))
    
    if not valid_texts_with_indices:
        return ["N/A"] * len(texts_list)

    original_indices, valid_texts_to_encode = zip(*valid_texts_with_indices)
    
    X_encoded = encode_texts_for_sentiment(list(valid_texts_to_encode), vocab, max_len, pad_idx, unk_idx)
    
    try:
        probs = model.predict(X_encoded)
        preds_binary = (probs > 0.5).astype(int)
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return ["Error: Prediction failed"] * len(texts_list)

    final_predictions = ["N/A"] * len(texts_list)
    sentiment_map = {1: "Positive", 0: "Negative"}

    for i, pred_binary_val in zip(original_indices, preds_binary):
        final_predictions[i] = sentiment_map[pred_binary_val.item()] 
        
    return final_predictions