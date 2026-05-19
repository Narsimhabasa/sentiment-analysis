# ============================================================
# Twitter Sentiment Analysis using NLP
# Author: Narsimha Basa | github.com/Narsimhabasa
# ============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── 2. LOAD DATA ─────────────────────────────────────────────
# Dataset: Sentiment140 (1.6M tweets) from Kaggle
# https://www.kaggle.com/datasets/kazanova/sentiment140
# Column layout: target, ids, date, flag, user, text
# For demo we simulate a small balanced sample

print("Loading dataset...")

# Simulated sample for demo (replace with actual CSV path)
np.random.seed(42)
n = 3000
positive_samples = [
    "I love this product so much!", "Great day today, feeling awesome!",
    "This is the best thing ever!", "So happy with the results!",
    "Amazing experience, highly recommend!", "Feeling blessed and grateful today",
    "Just got great news, so excited!", "This made my day so much better",
    "Absolutely loving every moment of this", "Fantastic service and great vibes",
] * (n // 20)
negative_samples = [
    "This is terrible, very disappointed.", "Worst experience I have ever had.",
    "I hate how this turned out.", "Feeling really down today.",
    "Such a waste of time and money.", "Nothing is going right today.",
    "Completely frustrated with this service.", "This product broke on the first day.",
    "So disappointed with the outcome.", "Terrible customer support, never again.",
] * (n // 20)
neutral_samples = [
    "I went to the store today.", "The weather is cloudy outside.",
    "Just finished reading a book.", "Had lunch at a new restaurant.",
    "Watching TV after work today.", "The meeting is scheduled for Monday.",
    "Ordered something online yesterday.", "The train was slightly delayed.",
    "Updated my phone software today.", "Replied to a few emails this morning.",
] * (n // 20)

texts = positive_samples[:1000] + negative_samples[:1000] + neutral_samples[:1000]
labels = ['positive'] * 1000 + ['negative'] * 1000 + ['neutral'] * 1000

df = pd.DataFrame({'text': texts, 'sentiment': labels}).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Dataset shape: {df.shape}")
print(df['sentiment'].value_counts())

# ── 3. TEXT PREPROCESSING ────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'@\w+', '', text)                  # remove mentions
    text = re.sub(r'#\w+', '', text)                  # remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)              # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()          # remove extra spaces
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

print("\nCleaning text...")
df['clean_text'] = df['text'].apply(clean_text)
print("Sample cleaned text:")
print(df[['text', 'clean_text', 'sentiment']].head(3).to_string())

# ── 4. TRAIN / TEST SPLIT ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── 5. MODEL: TF-IDF + LOGISTIC REGRESSION ───────────────────
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
])

print("\nTraining model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 6. VISUALIZATIONS ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Twitter Sentiment Analysis – Results', fontsize=15, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative', 'neutral'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['positive', 'negative', 'neutral'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix')

# Sentiment distribution
colors = ['#2ecc71', '#e74c3c', '#3498db']
df['sentiment'].value_counts().plot(kind='bar', ax=axes[1], color=colors, edgecolor='white')
axes[1].set_title('Sentiment Distribution in Dataset')
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('results/sentiment_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to results/sentiment_results.png")

# ── 7. PREDICT ON NEW TWEETS ─────────────────────────────────
def predict_sentiment(tweet):
    cleaned = clean_text(tweet)
    pred = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    confidence = dict(zip(classes, [f"{p:.1%}" for p in proba]))
    return pred, confidence

print("\n── Live Predictions ──")
test_tweets = [
    "I absolutely love this new feature, it's amazing!",
    "This is the worst update they have ever released.",
    "Just finished my homework for tomorrow.",
]
for tweet in test_tweets:
    pred, conf = predict_sentiment(tweet)
    print(f"Tweet: {tweet}")
    print(f"  → Sentiment: {pred.upper()} | Confidence: {conf}\n")
