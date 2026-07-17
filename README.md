# Real-Time Fake News Detection Software

**FakeShield** — An MSc research project implementing a hybrid deep learning system for automated fake news detection, combining TF-IDF feature extraction with Bidirectional LSTM architecture, real-time news analysis, and LLM-based verification.

**Author:** Avijit Bose  
**Programme:** M.Sc. Computer Science  
**Institution:** West Bengal State University  
**Supervisor:** Dr. Kaushik Roy

---

## Project Overview

This project investigates the effectiveness of Machine Learning, Deep Learning, and Hybrid Architectures for fake news detection, and extends the system toward real-time misinformation verification. The final system — FakeShield — provides an interactive web interface capable of analysing news articles, fetching live headlines, and comparing predictions against a locally-hosted Large Language Model (Phi-3).

The research evaluates five classification approaches: Logistic Regression, Naive Bayes, Support Vector Machine (SVM), a standalone BiLSTM model, and a novel Hybrid TF-IDF + BiLSTM architecture. The Hybrid Model achieves the highest overall performance with **97.84% accuracy** and **97.91% F1-score** on the WELFake benchmark dataset.

---

## Research Objectives

- Detect fake news articles using AI-based Natural Language Processing (NLP) approaches
- Compare traditional Machine Learning baselines against Deep Learning architectures
- Develop and evaluate a Hybrid Model combining TF-IDF statistical features with BiLSTM sequential modelling
- Analyse model behaviour, including overfitting, temporal bias, and distribution shift
- Build a real-time fake news detection framework with live news API integration
- Compare the Hybrid Model against a local Large Language Model (Phi-3 via Ollama)
- Provide a user-friendly, research-grade interface for prediction and analysis

---

## System Architecture

```
FakeNewsDetection/
├── config/
│   └── config.py               # API keys, constants, trusted sources
├── preprocessing/
│   └── text_cleaner.py         # Lowercasing, HTML removal, stopword filtering, lemmatisation
├── feature_engineering/
│   └── vectorizer.py           # TF-IDF vectoriser (50K features, bigrams)
├── training/
│   └── train_model.py          # ML baseline training (LR, NB, SVM)
├── notebooks/
│   ├── 02_BiLSTM_Model.ipynb   # BiLSTM training and evaluation
│   └── 03_Hybrid_Model.ipynb   # Hybrid TF-IDF + BiLSTM experiments
├── realtime/
│   ├── news_api.py             # NewsAPI integration
│   ├── rss_fetcher.py          # RSS feed fetcher (BBC, Reuters, Al Jazeera)
│   └── llm_analyzer.py         # Phi-3 LLM integration via Ollama REST API
├── credibility/
│   └── scorer.py               # Source credibility scoring
├── database/
│   └── db.py                   # SQLite prediction logging
├── gui/
│   └── app.py                  # Streamlit web application (6 tabs)
├── models/                     # Saved model files (.keras, .pkl)
├── data/                       # Dataset files (excluded from repo)
└── tests/
```

---

## Technologies

| Category | Libraries / Tools |
|---|---|
| Language | Python 3.10 |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| NLP | NLTK |
| Data Handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Web Interface | Streamlit |
| News APIs | NewsAPI, RSS (feedparser) |
| LLM Integration | Ollama (Phi-3 Mini, local) |
| Database | SQLite3 |

---

## Dataset

**WELFake Dataset**

The WELFake dataset is a merged benchmark combining four publicly available fake news datasets: Kaggle, McIntire, Reuters, and BuzzFeed Political.

| Property | Value |
|---|---|
| Total articles | 72,134 |
| Fake articles | 37,106 |
| Real articles | 35,028 |
| Class balance | Near-balanced (~51.4% fake) |
| Features used | Title + Content (combined) |
| Training split | 80% train / 20% test |

> Dataset files are excluded from this repository due to size constraints. The WELFake dataset is publicly available via Kaggle.

---

## Model Results

### Machine Learning Baselines

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 88.15% | 87.51% | 90.19% | 88.83% |
| Logistic Regression | 96.43% | 96.39% | 96.78% | 96.59% |
| SVM | 97.66% | 97.43% | 98.11% | 97.77% |

### Deep Learning Models

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| BiLSTM | 96.81% | 95.99% | 97.99% | 96.98% |
| **Hybrid v2 (TF-IDF + BiLSTM)** | **97.84%** | **97.83%** | **97.99%** | **97.91%** |

The Hybrid Model v2 was selected as the final model based on highest test-set F1-score. Hybrid v3 (with L2 regularisation) reduced overfitting but lowered accuracy to 95.87%, and was not selected.

## Experimental Results

### 🔹 Logistic Regression

![Logistic Regression](models/Logistic_Regression_confusion.png)

---

### 🔹 Naive Bayes

![Naive Bayes](models/Naive_Bayes_confusion.png)

---

### 🔹 Support Vector Machine (SVM)

![SVM](models/SVM_confusion.png)

---

### 🔹 BiLSTM Confusion Matrix

![BiLSTM](models/BiLSTM_confusion.png)

### Training Curve

![BiLSTM Curve](models/bilstm_training_curves.png)

---

### 🔹 Hybrid Model (TF-IDF + BiLSTM)

#### Confusion Matrix

![Hybrid](models/Hybrid_confusion.png)

#### Training Curve

![Hybrid Curve](models/hybrid_training_curves.png)

---

### 🔹 Hybrid Model v3

#### Confusion Matrix

![Hybrid V3](models/Hybrid_v3_confusion.png)

#### Training Curve

![Hybrid V3 Curve](models/hybrid_v3_training_curves.png)

---

## Key Research Findings

### 1. Hybrid Architecture Advantage

The TF-IDF + BiLSTM hybrid marginally outperforms standalone SVM and BiLSTM individually. The TF-IDF branch captures global keyword signals while the BiLSTM branch models sequential and contextual patterns, and their combination produces complementary features.

### 2. Overfitting Analysis

All Hybrid Model variants showed training-validation accuracy divergence (training ~99.9% vs. validation ~97.8%). Experiments with dropout, L2 regularisation, and GloVe embeddings consistently reduced overfitting but also reduced test accuracy. This behaviour is attributed to the WELFake dataset's limited stylistic diversity — the dataset is too "clean" and separable for regularisation to improve generalisation.

### 3. Temporal Distribution Shift

The model was trained on news data primarily from 2015–2018. Evaluation on live 2026 headlines revealed significant degradation, with approximately 89% of genuine live articles incorrectly classified as fake. This is attributed to style differences between Reuters-dominated WELFake real news and modern journalistic writing.

### 4. LLM Comparison Analysis

A qualitative comparison between the Hybrid Model and a local Phi-3 Mini LLM (via Ollama) revealed complementary strengths and weaknesses.

In one experiment using a recent BBC news article about Cape Verde's draw against Spain, the Hybrid Model correctly classified the article as **REAL** with **99.28% confidence**, while the local Phi-3 model incorrectly classified it as **FAKE** with **90% confidence**.

This result suggests that the supervised Hybrid Model generalized better than the local LLM for this particular article, whereas the LLM appeared to rely on imperfect reasoning despite understanding the article context. The disagreement also highlights that different AI approaches make different types of errors, reinforcing the importance of combining machine learning predictions with contextual analysis and source credibility verification.

## FakeShield — Web Application

The Streamlit-based GUI provides six tabs:

| Tab | Description |
|---|---|
| Text Analysis | Paste any article for instant prediction with confidence gauge, SVM cross-check, and probability bar |
| URL Analysis | Enter any news URL — the system fetches and analyses the content automatically |
| Model Comparison | Bar chart, radar chart, and full results table comparing all five models |
| Live News | Fetch and batch-analyse live headlines via NewsAPI or RSS feeds |
| LLM Compare | Side-by-side comparison of Hybrid Model vs. Phi-3 with agreement analysis and research insight |
| History | Prediction log with statistics, distribution charts, and CSV export |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Avijit0104/fake-news-detection-msc.git
cd fake-news-detection-msc
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Add API keys to `config/config.py`:

```python
NEWS_API_KEY  = "your_newsapi_key"     # https://newsapi.org/register
GEMINI_API_KEY = ""                    # optional — unused if using Phi-3
```

For LLM comparison, install Ollama and pull the Phi-3 model:

```bash
# Install from https://ollama.com
ollama pull phi3
ollama serve
```

Run the application:

```bash
streamlit run gui/app.py
```

---

## Model Files

Pre-trained model files required to run the application:

| File | Description |
|---|---|
| `models/hybrid_best_v2.keras` | Final Hybrid TF-IDF + BiLSTM model |
| `models/hybrid_tokenizer.pkl` | Keras tokeniser for Hybrid Model |
| `models/tfidf_hybrid.pkl` | TF-IDF vectoriser for Hybrid Model |
| `models/svm_model.pkl` | SVM baseline model |
| `models/tfidf_vectorizer.pkl` | TF-IDF vectoriser for SVM |

> Model files are excluded from this repository due to size. They can be reproduced by running the training notebooks in order.

---

## Limitations

- The model was trained on 2015–2018 news data and may misclassify articles written in different styles or covering post-2018 events.
- WELFake's real news portion is dominated by Reuters-style formal writing, which biases the model against emotionally-written genuine journalism.
- The Phi-3 LLM comparison relies on a locally-hosted model and requires Ollama to be running.
- Source credibility scoring uses a static list of trusted domains and does not dynamically verify domain reputation.

---

## Future Work

- Fine-tuning on a temporally diverse dataset to reduce distribution shift
- Integration of BERT or RoBERTa for contextual embeddings
- Dynamic source credibility scoring via third-party APIs
- Explainable AI (XAI) integration to provide article-level feature attribution
- Model deployment to Streamlit Cloud or a containerised environment
- Social media verification module

---

## Repository Structure Notes

Training notebooks must be run in order:
1. `preprocessing/text_cleaner.py` — run first to generate `data/cleaned_data.csv`
2. `feature_engineering/vectorizer.py` — generates `models/tfidf_vectorizer.pkl`
3. `training/train_model.py` — trains ML baselines
4. `notebooks/02_BiLSTM_Model.ipynb` — trains BiLSTM
5. `notebooks/03_Hybrid_Model.ipynb` — trains Hybrid Model v2 (final)

---

## License

This project is developed for academic and research purposes under West Bengal State University.