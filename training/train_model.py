import pandas as pd
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)
from feature_engineering.vectorizer import (build_tfidf_vectorizer,
                                            transform_text,
                                            save_vectorizer)
import matplotlib.pyplot as plt
import seaborn as sns

# Define BASE_DIR globally so all functions can use it
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'cleaned_data.csv'))
    df = df.dropna(subset=['cleaned_text'])
    return train_test_split(
        df['cleaned_text'], df['label'],
        test_size=0.2, random_state=42
    )


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"\n{'='*45}")
    print(f"  Model: {name}")
    print(f"{'='*45}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred,
                                target_names=['Real', 'Fake']))
    return {'Model': name, 'Accuracy': acc,
            'Precision': prec, 'Recall': rec, 'F1': f1}


def plot_confusion_matrix(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix — {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'models',
                f'{name.replace(" ", "_")}_confusion.png'))
    plt.show()


def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved → {path}")


if __name__ == "__main__":

    # Step 1: Load data FIRST
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    # Step 2: Build vectorizer fresh (avoids version mismatch)
    print("Building vectorizer...")
    tfidf = build_tfidf_vectorizer(X_train)
    save_vectorizer(tfidf, os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'))
    X_train_v = transform_text(tfidf, X_train)
    X_test_v  = transform_text(tfidf, X_test)
    print("Vectorizing done!")

    results = []

    # Model 1: Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_v, y_train)
    results.append(evaluate_model("Logistic Regression", lr, X_test_v, y_test))
    plot_confusion_matrix("Logistic Regression", lr, X_test_v, y_test)
    save_model(lr, os.path.join(BASE_DIR, 'models', 'logistic_regression.pkl'))

    # Model 2: Naive Bayes
    print("\nTraining Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train_v, y_train)
    results.append(evaluate_model("Naive Bayes", nb, X_test_v, y_test))
    plot_confusion_matrix("Naive Bayes", nb, X_test_v, y_test)
    save_model(nb, os.path.join(BASE_DIR, 'models', 'naive_bayes.pkl'))

    # Model 3: SVM
    print("\nTraining SVM...")
    svm = LinearSVC(max_iter=2000)
    svm.fit(X_train_v, y_train)
    results.append(evaluate_model("SVM", svm, X_test_v, y_test))
    plot_confusion_matrix("SVM", svm, X_test_v, y_test)
    save_model(svm, os.path.join(BASE_DIR, 'models', 'svm_model.pkl'))

    # Comparison Table
    print("\n")
    print("="*55)
    print("      ML MODELS COMPARISON TABLE")
    print("="*55)
    results_df = pd.DataFrame(results)
    results_df = results_df.round(4)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(BASE_DIR, 'models', 'ml_results.csv'), index=False)
    print("\nResults saved → models/ml_results.csv")