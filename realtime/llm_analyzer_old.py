import requests
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import GEMINI_API_KEY

# NO google.generativeai import — uses REST API directly to avoid protobuf conflict

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

def analyze_with_gemini(title, content=''):
    prompt = f"""You are an expert fake news detection system.
Analyze the following news article and determine if it is FAKE or REAL news.

Article Title: {title}
Article Content: {content[:1000] if content else 'Not provided'}

Respond ONLY with a valid JSON object in exactly this format (no markdown, no backticks):
{{
  "label": "FAKE" or "REAL",
  "confidence": <number between 50 and 99>,
  "explanation": "<one sentence explanation, max 150 characters>",
  "red_flags": ["<flag1>", "<flag2>"] or [],
  "verdict": "<FAKE NEWS DETECTED> or <APPEARS CREDIBLE>"
}}

Rules:
- label must be exactly FAKE or REAL
- confidence is your certainty percentage (50-99)
- explanation is brief and factual
- red_flags lists specific issues if FAKE, empty list if REAL"""

    try:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature"    : 0.1,
                "maxOutputTokens": 300,
            }
        }

        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers = {'Content-Type': 'application/json'},
            json    = payload,
            timeout = 30
        )

        if response.status_code != 200:
            return {
                'label'      : 'ERROR',
                'confidence' : 0,
                'explanation': f"API Error {response.status_code}",
                'red_flags'  : [],
                'verdict'    : '',
                'error'      : response.text[:200]
            }

        data     = response.json()
        raw_text = data['candidates'][0]['content']['parts'][0]['text']
        raw_text = raw_text.replace('```json','').replace('```','').strip()

        result = json.loads(raw_text)
        return {
            'label'      : str(result.get('label','UNKNOWN')).upper(),
            'confidence' : int(result.get('confidence', 70)),
            'explanation': result.get('explanation', 'Analysis complete.'),
            'red_flags'  : result.get('red_flags', []),
            'verdict'    : result.get('verdict', ''),
            'error'      : None
        }

    except json.JSONDecodeError:
        label = 'FAKE' if 'FAKE' in raw_text.upper() else 'REAL'
        return {
            'label'      : label,
            'confidence' : 70,
            'explanation': 'Analysis completed.',
            'red_flags'  : [],
            'verdict'    : '',
            'error'      : None
        }
    except Exception as e:
        return {
            'label'      : 'ERROR',
            'confidence' : 0,
            'explanation': str(e)[:100],
            'red_flags'  : [],
            'verdict'    : '',
            'error'      : str(e)
        }
    
    if response.status_code != 200:
        if response.status_code == 429:
            return {
                'label'      : 'ERROR',
                'confidence' : 0,
                'explanation': 'Gemini rate limit reached. Please wait 30 seconds and try again.',
                'red_flags'  : [],
                'verdict'    : '',
                'error'      : 'Rate limit (429) — wait 30 seconds'
            }
        return {
            'label'      : 'ERROR',
            'confidence' : 0,
            'explanation': f"API Error {response.status_code}",
            'red_flags'  : [],
            'verdict'    : '',
            'error'      : response.text[:200]
        }


def get_agreement_analysis(model_label, gemini_label, model_conf, gemini_conf):
    if model_label == gemini_label:
        avg_conf = round((model_conf + gemini_conf) / 2, 1)
        return {
            'status'     : 'AGREE',
            'color'      : '#38ef7d',
            'icon'       : '✅',
            'message'    : f'Both models agree — {model_label}',
            'avg_conf'   : avg_conf,
            'reliability': 'HIGH' if avg_conf > 80 else 'MEDIUM'
        }
    else:
        return {
            'status'     : 'DISAGREE',
            'color'      : '#ffd700',
            'icon'       : '⚠️',
            'message'    : 'Models disagree — manual verification recommended',
            'avg_conf'   : round((model_conf + gemini_conf) / 2, 1),
            'reliability': 'LOW'
        }