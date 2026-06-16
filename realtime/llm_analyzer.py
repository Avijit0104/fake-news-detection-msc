import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"


def analyze_with_phi3(title, content=""):
    prompt = f"""You are a fake news detection expert.
Analyze this news article and respond ONLY with valid JSON, no extra text.

Title: {title}
Content: {content[:800] if content else 'Not provided'}

Respond with exactly this JSON format:
{{"label": "FAKE", "confidence": 85, "explanation": "one sentence reason", "red_flags": ["flag1"]}}

label must be FAKE or REAL. confidence between 50-99."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model"  : "phi3",
                "prompt" : prompt,
                "stream" : False
            },
            timeout=60
        )

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
            label = 'FAKE' if 'FAKE' in raw.upper() else 'REAL'
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