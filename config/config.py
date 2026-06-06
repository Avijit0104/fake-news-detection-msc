<<<<<<< HEAD
=======
import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# News categories available
NEWS_CATEGORIES = [
    "general", "technology", "science",
    "health", "business", "sports", "entertainment"
]

# Trusted news sources for credibility scoring
TRUSTED_SOURCES = [
    'bbc.com', 'bbc.co.uk', 'reuters.com', 'apnews.com',
    'theguardian.com', 'nytimes.com', 'washingtonpost.com',
    'bloomberg.com', 'forbes.com', 'hindustantimes.com',
    'thehindu.com', 'ndtv.com', 'timesofindia.com',
    'aljazeera.com', 'cnbc.com', 'cnn.com', 'thewire.in'
]

MEDIUM_SOURCES = [
    'yahoo.com', 'msn.com', 'huffpost.com',
    'buzzfeed.com', 'dailymail.co.uk', 'foxnews.com'
]

# Model settings
MAX_LEN        = 300
MAX_TFIDF      = 10000
THRESHOLD      = 0.72
>>>>>>> e004906 (changed llm)
