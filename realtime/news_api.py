import requests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import NEWS_API_KEY, TRUSTED_SOURCES, MEDIUM_SOURCES


def fetch_live_news(category='general', country='us', page_size=10):
    """
    Fetch live news headlines from NewsAPI.
    Returns list of article dicts.
    """
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        'apiKey'   : NEWS_API_KEY,
        'category' : category,
        'language' : 'en',
        'pageSize' : page_size,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data     = response.json()

        if data.get('status') != 'ok':
            return [], f"API Error: {data.get('message', 'Unknown error')}"

        articles = []
        for a in data.get('articles', []):
            # Skip articles with no content
            if not a.get('title') or not a.get('description'):
                continue
            if a['title'] == '[Removed]':
                continue

            articles.append({
                'title'      : a.get('title', ''),
                'description': a.get('description', ''),
                'content'    : a.get('content', ''),
                'url'        : a.get('url', ''),
                'source'     : a.get('source', {}).get('name', 'Unknown'),
                'publishedAt': a.get('publishedAt', ''),
                'urlToImage' : a.get('urlToImage', ''),
            })

        return articles, None

    except requests.exceptions.ConnectionError:
        return [], "No internet connection"
    except requests.exceptions.Timeout:
        return [], "Request timed out"
    except Exception as e:
        return [], str(e)


def search_news(query, page_size=10):
    """Search news by keyword"""
    url    = "https://newsapi.org/v2/everything"
    params = {
        'apiKey'   : NEWS_API_KEY,
        'q'        : query,
        'language' : 'en',
        'sortBy'   : 'publishedAt',
        'pageSize' : page_size,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data     = response.json()

        if data.get('status') != 'ok':
            return [], f"API Error: {data.get('message', 'Unknown error')}"

        articles = []
        for a in data.get('articles', []):
            if not a.get('title') or a['title'] == '[Removed]':
                continue
            articles.append({
                'title'      : a.get('title', ''),
                'description': a.get('description', ''),
                'content'    : a.get('content', ''),
                'url'        : a.get('url', ''),
                'source'     : a.get('source', {}).get('name', 'Unknown'),
                'publishedAt': a.get('publishedAt', ''),
                'urlToImage' : a.get('urlToImage', ''),
            })

        return articles, None

    except Exception as e:
        return [], str(e)


def get_source_credibility(url='', source_name=''):
    """Score source credibility based on known trusted domains"""
    text = (url + source_name).lower()

    for trusted in TRUSTED_SOURCES:
        if trusted in text:
            return {
                'label': 'High Credibility',
                'score': 'HIGH',
                'color': '#38ef7d',
                'icon' : '✅'
            }

    for medium in MEDIUM_SOURCES:
        if medium in text:
            return {
                'label': 'Medium Credibility',
                'score': 'MEDIUM',
                'color': '#ffd700',
                'icon' : '⚠️'
            }

    return {
        'label': 'Unknown Source',
        'score': 'LOW',
        'color': '#ff416c',
        'icon' : '❓'
    }


def format_date(date_str):
    """Format ISO date to readable string"""
    try:
        from datetime import datetime
        dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
        return dt.strftime('%b %d, %Y %I:%M %p')
    except:
        return date_str