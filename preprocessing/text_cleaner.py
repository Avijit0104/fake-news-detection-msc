import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only runs once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Full NLP cleaning pipeline:
    1. Lowercase
    2. Remove HTML tags
    3. Remove URLs
    4. Remove punctuation & special characters
    5. Remove extra whitespace
    6. Remove stopwords
    7. Lemmatize words
    """

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Step 3: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 4: Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Step 5: Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 6 & 7: Tokenize, remove stopwords, lemmatize
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) 
              for word in tokens 
              if word not in stop_words and len(word) > 2]

    return ' '.join(tokens)