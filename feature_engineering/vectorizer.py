from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def build_tfidf_vectorizer(X_train, max_features=50000, ngram_range=(1, 2)):
    """
    Build and fit TF-IDF vectorizer on training data.
    
    max_features=50000 means we keep the 50,000 most important words
    ngram_range=(1,2) means we capture single words AND pairs of words
    Example: "fake news" as a phrase, not just "fake" and "news" separately
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True  # applies log scaling — helps with very long articles
    )
    
    vectorizer.fit(X_train)
    return vectorizer


def transform_text(vectorizer, X):
    """Transform text data using fitted vectorizer"""
    return vectorizer.transform(X)


def save_vectorizer(vectorizer, path='../models/tfidf_vectorizer.pkl'):
    """Save vectorizer so we can reuse it later without retraining"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {path}")


def load_vectorizer(path='../models/tfidf_vectorizer.pkl'):
    """Load a previously saved vectorizer"""
    with open(path, 'rb') as f:
        return pickle.load(f)