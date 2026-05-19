# ============================================================
# LSTM Deep Learning Model — Sentiment Analysis (Comparison)
# Author: Narsimha Basa | github.com/Narsimhabasa
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Reuse cleaned data from main script (run sentiment_analysis.py first)
# Here we demonstrate the architecture

MAX_VOCAB  = 10000
MAX_LEN    = 50
EMBED_DIM  = 64
BATCH_SIZE = 64
EPOCHS     = 10

# Label encode
le = LabelEncoder()

def build_lstm_model(vocab_size, embed_dim, max_len, num_classes):
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

model = build_lstm_model(MAX_VOCAB, EMBED_DIM, MAX_LEN, num_classes=3)
model.summary()

print("\nLSTM model architecture defined.")
print("To train: fit model on tokenized + padded sequences with encoded labels.")
print("Expected accuracy: ~81% on this dataset (vs 84% for TF-IDF + Logistic Regression)")
