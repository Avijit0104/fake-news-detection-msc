import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"


def analyze_with_phi3(title, content=""):
    prompt = f"""
You are a professional fact-checking assistant.

Analyze the article carefully.

Rules:
1. Do NOT guess facts.
2. If the article appears factual and contains no obvious misinformation, return REAL.
3. If the article contains fabricated claims, contradictions, or misinformation, return FAKE.
4. If there is insufficient evidence from the article alone, return UNCERTAIN.
5. Use conservative reasoning.

Title:
{title}

Content:
{content[:1200] if content else "Not provided"}

Return ONLY valid JSON:

{{
    "label": "REAL",
    "confidence": 85,
    "explanation": "Brief explanation",
    "red_flags": []
}}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model"  : "llama3.1:8b",
                "prompt" : prompt,
                "stream" : False
            },
            timeout=180
        )



        if response.status_code != 200:
            return {
                'label': 'ERROR',
                'confidence': 0,
                'explanation': f'Ollama error {response.status_code}',
                'red_flags': [],
                'verdict': '',
                'error': response.text[:200]
            }
        raw = response.json().get("response", "")

        # Clean markdown fences if present
        raw = raw.replace('```json','').replace('```','').strip()

        # Extract JSON from response
        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                'label'      : str(result.get('label','REAL')).upper(),
                'confidence' : int(result.get('confidence', 75)),
                'explanation': result.get('explanation', 'Analysis complete.'),
                'red_flags'  : result.get('red_flags', []),
                'verdict'    : '',
                'error'      : None
            }
        else:
            # Fallback — extract label from raw text
            raw_upper = raw.upper()

            if "UNCERTAIN" in raw_upper:
                label = "UNCERTAIN"
            elif "FAKE" in raw_upper:
                label = "FAKE"
            else:
                label = "REAL"
            return {
                'label'      : label,
                'confidence' : 70,
                'explanation': raw[:150].strip(),
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


def get_agreement_analysis(model_label, llm_label, model_conf, llm_conf):
    avg_conf = round((model_conf + llm_conf) / 2, 1)
    if model_label == llm_label:
        return {
            'status'     : 'AGREE',
            'color'      : '#38ef7d',
            'icon'       : '✅',
            'message'    : f'Both models agree — {model_label}',
            'avg_conf'   : avg_conf,
            'reliability': 'HIGH' if avg_conf > 80 else 'MEDIUM'
        }
    return {
        'status'     : 'DISAGREE',
        'color'      : '#ffd700',
        'icon'       : '⚠️',
        'message'    : 'Models disagree — manual verification recommended',
        'avg_conf'   : avg_conf,
        'reliability': 'LOW'
    }