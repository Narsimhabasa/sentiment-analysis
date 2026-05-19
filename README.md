# 🐦 Twitter Sentiment Analysis using NLP

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-84%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

> Classifying tweets as **Positive**, **Negative**, or **Neutral** using TF-IDF vectorization and Logistic Regression — with an LSTM deep learning comparison.

---

## 📌 Project Overview

This project builds a complete NLP pipeline to perform sentiment analysis on Twitter data. It covers everything from raw text cleaning to model evaluation and live predictions.

| Item | Detail |
|---|---|
| Dataset | Sentiment140 — 50,000+ tweets |
| Task | 3-class classification (Positive / Negative / Neutral) |
| Best Model | Logistic Regression + TF-IDF |
| Accuracy | **84%** |
| Deep Learning | LSTM comparison included |

---

## 🗂️ Project Structure

```
sentiment-analysis/
│
├── sentiment_analysis.py     # Main ML pipeline (TF-IDF + Logistic Regression)
├── lstm_model.py             # LSTM deep learning model (comparison)
├── requirements.txt          # Python dependencies
├── results/
│   └── sentiment_results.png # Confusion matrix & distribution plots
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Narsimhabasa/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the main script
```bash
python sentiment_analysis.py
```

---

## 🔍 Methodology

### Text Preprocessing
- Lowercasing, URL/mention/hashtag removal
- Punctuation and special character removal
- Stopword removal using NLTK
- Lemmatization using WordNetLemmatizer

### Feature Engineering
- **TF-IDF Vectorizer** with unigrams + bigrams (`ngram_range=(1,2)`)
- Max features: 10,000

### Models Compared
| Model | Accuracy |
|---|---|
| Logistic Regression + TF-IDF | **84%** |
| LSTM (Deep Learning) | ~81% |

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization

---

## 📊 Results

![Sentiment Results](results/sentiment_results.png)

**Key findings:**
- The model performs best on clearly positive/negative tweets
- Neutral tweets are harder to classify (common in NLP tasks)
- Bigrams significantly improved accuracy over unigrams alone

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas, NumPy** — data manipulation
- **NLTK** — text preprocessing
- **Scikit-learn** — TF-IDF, Logistic Regression, evaluation
- **Matplotlib, Seaborn** — visualizations
- **TensorFlow/Keras** — LSTM model

---

## 👤 Author

**Narsimha Basa**
- GitHub: [@Narsimhabasa](https://github.com/Narsimhabasa)
- Email: narsimhabasa1206@gmail.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
