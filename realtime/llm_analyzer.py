import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"


def analyze_with_phi3(title, content=""):

    prompt = f"""
You are a fake news detection expert.

Analyze this article.

Title:
{title}

Content:
{content[:1000]}

Return ONLY JSON:

{{
    "label": "FAKE or REAL",
    "confidence": 50-99,
    "explanation": "short explanation"
}}
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        data = response.json()

        raw = data["response"]

        return {
            "label": "REAL" if "REAL" in raw.upper() else "FAKE",
            "confidence": 80,
            "explanation": raw[:200]
        }

    except Exception as e:
        return {
            "label": "ERROR",
            "confidence": 0,
            "explanation": str(e)
        }
def get_agreement_analysis(model_label,
                           llm_label,
                           model_conf,
                           llm_conf):

    if model_label == llm_label:
        avg_conf = round((model_conf + llm_conf) / 2, 1)

        return {
            "status": "AGREE",
            "color": "#38ef7d",
            "icon": "✅",
            "message": f"Both models agree — {model_label}",
            "avg_conf": avg_conf,
            "reliability": "HIGH" if avg_conf > 80 else "MEDIUM"
        }

    avg_conf = round((model_conf + llm_conf) / 2, 1)

    return {
        "status": "DISAGREE",
        "color": "#ffd700",
        "icon": "⚠️",
        "message": "Models disagree — manual verification recommended",
        "avg_conf": avg_conf,
        "reliability": "LOW"
    }