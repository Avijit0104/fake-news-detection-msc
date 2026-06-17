import streamlit as st
import sys
import os
import pickle
import numpy as np
import time
import plotly.graph_objects as go
import html as html_lib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessing.text_cleaner import clean_text

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="FakeShield — Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .main-header { text-align: center; padding: 2rem 0 1rem 0; }
    .main-title {
        font-size: 3.5rem; font-weight: 900;
        background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
    }
    .main-subtitle { font-size: 1.1rem; color: #aaa; margin-top: -10px; }
    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important; color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 12px !important; font-size: 0.95rem !important;
    }
    .stTextInput input {
        background: rgba(255,255,255,0.05) !important; color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7b2ff7, #00d2ff);
        color: white; border: none; border-radius: 12px;
        padding: 0.75rem 1.5rem; font-size: 1rem; font-weight: 700;
        letter-spacing: 1px; transition: all 0.3s ease; cursor: pointer;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(123,47,247,0.5);
    }
    .result-fake {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 16px; padding: 2rem; text-align: center;
        box-shadow: 0 8px 32px rgba(255,65,108,0.4);
    }
    .result-real {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px; padding: 2rem; text-align: center;
        box-shadow: 0 8px 32px rgba(56,239,125,0.4);
    }
    .result-label { font-size: 3rem; font-weight: 900; color: white; letter-spacing: 3px; }
    .result-emoji { font-size: 4rem; }
    .metric-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 1.2rem; text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #00d2ff; }
    .metric-label { font-size: 0.8rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stSidebar"] {
        background: rgba(15,12,41,0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.05); border-radius: 12px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] { color: #aaa; border-radius: 8px; font-weight: 600; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7b2ff7, #00d2ff) !important;
        color: white !important;
    }
    hr { border-color: rgba(255,255,255,0.1); }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── LOAD MODELS ───────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

@st.cache_resource
def load_all_models():
    models = {}
    try:
        models['hybrid'] = load_model(os.path.join(BASE_DIR, 'models', 'hybrid_best_v2.keras'))
        with open(os.path.join(BASE_DIR, 'models', 'hybrid_tokenizer.pkl'), 'rb') as f:
            models['tokenizer'] = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'models', 'tfidf_hybrid.pkl'), 'rb') as f:
            models['tfidf'] = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'models', 'svm_model.pkl'), 'rb') as f:
            models['svm'] = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.pkl'), 'rb') as f:
            models['svm_tfidf'] = pickle.load(f)
        models['loaded'] = True
    except Exception as e:
        models['loaded'] = False
        models['error']  = str(e)
    return models

models = load_all_models()


# ── CONSTANTS & HELPERS ───────────────────────────────────────
MAX_LEN   = 300
MAX_TFIDF = 10000

TRUSTED_SOURCES = [
    'bbc.com', 'bbc.co.uk', 'reuters.com', 'apnews.com', 'theguardian.com',
    'nytimes.com', 'washingtonpost.com', 'bloomberg.com', 'forbes.com',
    'hindustantimes.com', 'thehindu.com', 'ndtv.com', 'timesofindia.com',
    'aljazeera.com', 'cnbc.com', 'cnn.com', 'thewire.in'
]


def get_source_credibility(url='', source_name=''):
    text = (url + source_name).lower()
    if not text.strip():
        return {'label': 'Unknown Source', 'color': '#888888', 'icon': '❓'}
    for source in TRUSTED_SOURCES:
        if source in text:
            return {'label': 'Trusted Source', 'color': '#38ef7d', 'icon': '✅'}
    return {'label': 'Unknown Source', 'color': '#ffd700', 'icon': '⚠️'}

def predict_with_ensemble(text, url='', phi3_label=None):
    """
    Combine Hybrid Model + Phi-3 for smarter prediction.
    Only mark FAKE when both models agree OR confidence is very high.
    """
    result = predict_news(text, url)

    if phi3_label is None:
        return result  # no LLM available, use model alone

    model_label = result['label']
    model_conf  = result['confidence']

    # Both agree → high reliability
    if model_label == phi3_label:
        result['final_label']  = model_label
        result['reliability']  = 'HIGH'
        result['verdict_note'] = f'Both models agree: {model_label}'

    # Only your model says FAKE with low confidence → UNCERTAIN
    elif model_label == 'FAKE' and model_conf < 90:
        result['final_label']  = 'UNCERTAIN'
        result['reliability']  = 'LOW'
        result['verdict_note'] = 'Models disagree — manual verification needed'

    # Phi-3 says FAKE but your model says REAL → trust phi-3
    elif phi3_label == 'FAKE' and model_label == 'REAL':
        result['final_label']  = 'UNCERTAIN'
        result['reliability']  = 'MEDIUM'
        result['verdict_note'] = 'LLM flagged potential misinformation'

    else:
        result['final_label']  = model_label
        result['reliability']  = 'MEDIUM'
        result['verdict_note'] = 'Based on hybrid model analysis'

    return result



def predict_news(text, url=''):
    start_time = time.time()
    cleaned    = clean_text(text)

    seq     = models['tokenizer'].texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    tfidf_v = models['tfidf'].transform([cleaned]).toarray().astype('float32')
    prob    = models['hybrid'].predict([padded, tfidf_v], verbose=0)[0][0]

    label = 'FAKE' if prob > 0.85 else 'REAL'
    conf  = prob if prob > 0.85 else 1 - prob

    svm_v     = models['svm_tfidf'].transform([cleaned])
    svm_pred  = models['svm'].predict(svm_v)[0]
    svm_label = 'FAKE' if svm_pred == 1 else 'REAL'

    latency = round(time.time() - start_time, 3)
    cred    = get_source_credibility(url)

    return {
        'label'            : label,
        'confidence'       : round(float(conf) * 100, 2),
        'probability'      : round(float(prob) * 100, 2),
        'svm_label'        : svm_label,
        'agreement'        : label == svm_label,
        'latency'          : latency,
        'cleaned'          : cleaned,
        'credibility'      : cred['label'],
        'credibility_color': cred['color'],
        'credibility_icon' : cred['icon'],
    }


def render_gauge(confidence, label):
    color = '#ff416c' if label == 'FAKE' else '#38ef7d'
    fig   = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = confidence,
        title = {'text': "Confidence %", 'font': {'color': 'white', 'size': 14}},
        number= {'suffix': "%", 'font': {'color': color, 'size': 28}},
        gauge = {
            'axis'     : {'range': [0, 100], 'tickcolor': 'white', 'tickfont': {'color': 'white'}},
            'bar'      : {'color': color},
            'bgcolor'  : 'rgba(0,0,0,0)',
            'steps'    : [
                {'range': [0,  50], 'color': 'rgba(255,255,255,0.05)'},
                {'range': [50, 75], 'color': 'rgba(255,200,0,0.1)'},
                {'range': [75,100], 'color': 'rgba(255,65,108,0.1)'},
            ],
            'threshold': {'line': {'color': color, 'width': 4}, 'thickness': 0.75, 'value': confidence}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', height=250, margin=dict(t=40, b=0, l=20, r=20)
    )
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:3rem;'>🛡️</div>
        <div style='font-size:1.3rem;font-weight:800;
                    background:linear-gradient(90deg,#00d2ff,#7b2ff7);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            FakeShield
        </div>
        <div style='color:#aaa;font-size:0.8rem;'>AI-Powered Fake News Detector</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    for name, value in [("Hybrid Model","97.84%"),("SVM Baseline","97.66%"),
                         ("F1 Score","97.91%"),("Dataset","72K Articles")]:
        st.markdown(f"""
        <div class='metric-card' style='margin-bottom:8px;'>
            <div class='metric-value'>{value}</div>
            <div class='metric-label'>{name}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style='color:#aaa;font-size:0.85rem;line-height:1.6;'>
    Built by <b style='color:white'>Avijit Bose</b><br>
    MSc Computer Science<br>West Bengal State University<br><br>
    <b style='color:white'>Model:</b> TF-IDF + BiLSTM<br>
    <b style='color:white'>Dataset:</b> WELFake (72K)
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='background:rgba(255,200,0,0.08);border:1px solid rgba(255,200,0,0.2);
    border-radius:10px;padding:0.8rem;'>
    <div style='color:#ffd700;font-weight:700;font-size:0.85rem;margin-bottom:6px;'>
    ⚠️ Model Limitations</div>
    <div style='color:#888;font-size:0.78rem;line-height:1.7;'>
    • Trained on 2015–2018 news data<br>
    • Best on WELFake-style articles<br>
    • May misclassify post-2018 news<br>
    • Always verify with trusted sources
    </div></div>""", unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <div class='main-title'>🛡️ FAKESHIELD</div>
    <div class='main-subtitle'>Real-Time Fake News Detection powered by AI & NLP</div>
</div>""", unsafe_allow_html=True)
st.markdown("---")

if not models.get('loaded'):
    st.error(f"❌ Models failed to load: {models.get('error')}")
    st.stop()


# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝  Text Analysis",
    "🔗  URL Analysis",
    "📊  Model Comparison",
    "📡  Live News",
    "🤖  LLM Compare"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — TEXT ANALYSIS
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter News Article")
    col_input, col_tip = st.columns([3, 1])

    with col_input:
        headline = st.text_input("📰 Headline (optional)", placeholder="Enter the news headline here...")
        content  = st.text_area("📄 Article Content", placeholder="Paste the full news article content here...", height=200)

    with col_tip:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.05);border-radius:12px;padding:1rem;
                    border:1px solid rgba(255,255,255,0.1);margin-top:1.8rem;'>
            <div style='color:#00d2ff;font-weight:700;margin-bottom:8px;'>💡 Tips</div>
            <div style='color:#aaa;font-size:0.85rem;line-height:1.8;'>
            ✅ Paste full article<br>✅ Include headline<br>✅ Min 50 words<br>✅ English text only
            </div>
        </div>""", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        analyze_btn = st.button("🔍 ANALYZE", key="analyze_text")
    with col_btn2:
        st.button("🗑️ CLEAR", key="clear_text")

    if analyze_btn:
        full_text = f"{headline} {content}".strip()
        if len(full_text.split()) < 10:
            st.warning("⚠️ Please enter at least 10 words for accurate analysis.")
        else:
            with st.spinner("🤖 Analyzing with AI..."):
                # result = predict_news(full_text)
                result = predict_with_ensemble(full_text,phi3_label=phi3_label)
                time.sleep(0.5)

            st.markdown("---")
            st.markdown("### 🎯 Analysis Results")
            col_res, col_gauge = st.columns([1, 1])

            with col_res:
                final_label = result.get('final_label', result['label'])

                if final_label == 'FAKE':
                    st.markdown("""
                    <div class='result-fake'>
                        <div class='result-emoji'>🚨</div>
                        <div class='result-label'>FAKE NEWS</div>
                        <div style='color:rgba(255,255,255,0.8);margin-top:8px;font-size:0.95rem;'>
                        This article shows signs of misinformation</div>
                    </div>""", unsafe_allow_html=True)

                elif final_label == 'REAL':
                    st.markdown("""
                    <div class='result-real'>
                        <div class='result-emoji'>✅</div>
                        <div class='result-label'>REAL NEWS</div>
                        <div style='color:rgba(255,255,255,0.8);margin-top:8px;font-size:0.95rem;'>
                        This article appears to be credible</div>
                    </div>""", unsafe_allow_html=True)

                else:  # UNCERTAIN
                    st.markdown("""
                    <div style='background:linear-gradient(135deg,#f7971e,#ffd200);
                                border-radius:16px;padding:2rem;text-align:center;
                                box-shadow:0 8px 32px rgba(255,210,0,0.4);'>
                        <div style='font-size:4rem;'>⚠️</div>
                        <div style='font-size:3rem;font-weight:900;color:white;letter-spacing:3px;'>
                        UNCERTAIN</div>
                        <div style='color:rgba(255,255,255,0.9);margin-top:8px;font-size:0.95rem;'>
                        Models disagree — manual verification recommended</div>
                    </div>""", unsafe_allow_html=True)

            with col_gauge:
                st.plotly_chart(render_gauge(result['confidence'], result['label']), use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            conf_color  = '#ff416c' if result['label'] == 'FAKE' else '#38ef7d'
            agree_color = '#38ef7d' if result['agreement'] else '#ff416c'
            agree_text  = 'Agree ✅' if result['agreement'] else 'Disagree ⚠️'

            with m1:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{conf_color};'>{result['confidence']}%</div><div class='metric-label'>Confidence</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{agree_color};font-size:1.2rem;'>{result['svm_label']}</div><div class='metric-label'>SVM Says</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{agree_color};'>{agree_text}</div><div class='metric-label'>Model Agreement</div></div>", unsafe_allow_html=True)
            with m4:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{result['latency']}s</div><div class='metric-label'>Response Time</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📈 Fake vs Real Probability")
            fake_prob = result['probability']
            real_prob = round(100 - fake_prob, 2)
            st.markdown(f"""
            <div style='margin:8px 0;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                    <span style='color:#ff416c;font-weight:700;'>🔴 FAKE {fake_prob}%</span>
                    <span style='color:#38ef7d;font-weight:700;'>✅ REAL {real_prob}%</span>
                </div>
                <div style='background:rgba(255,255,255,0.1);border-radius:10px;height:20px;overflow:hidden;'>
                    <div style='width:{fake_prob}%;height:100%;background:linear-gradient(90deg,#ff416c,#ff4b2b);border-radius:10px;'></div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background:rgba(255,200,0,0.1);border:1px solid rgba(255,200,0,0.3);
                        border-radius:12px;padding:1rem;margin-top:12px;'>
                <span style='color:#ffd700;font-weight:700;'>⚠️ Important Note</span>
                <div style='color:#aaa;font-size:0.85rem;margin-top:6px;'>
                This model was trained on news data from 2015–2018. Accuracy may vary for recent
                news articles. Always cross-check with trusted news sources.
                </div>
            </div>""", unsafe_allow_html=True)

            if result['label'] == 'FAKE' and result['confidence'] < 85:
                st.markdown("""
                <div style='background:rgba(255,200,0,0.08);border:1px solid rgba(255,200,0,0.25);
                            border-radius:12px;padding:1rem;margin-top:1rem;'>
                    <span style='color:#ffd700;font-weight:700;font-size:0.95rem;'>⚠️ Low Confidence Warning</span>
                    <div style='color:#aaa;font-size:0.85rem;margin-top:6px;line-height:1.7;'>
                    Confidence below 85% — result may be unreliable. Always verify with trusted sources.
                    </div>
                </div>""", unsafe_allow_html=True)

            with st.expander("🔍 View Cleaned & Processed Text"):
                st.markdown(f"""
                <div style='background:rgba(0,0,0,0.3);border-radius:8px;padding:1rem;
                            color:#aaa;font-size:0.85rem;line-height:1.8;font-family:monospace;'>
                    {result['cleaned'][:500]}...
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — URL ANALYSIS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Analyze News from URL")
    st.markdown("""
    <div style='background:rgba(0,210,255,0.1);border:1px solid rgba(0,210,255,0.3);
                border-radius:12px;padding:1rem;margin-bottom:1rem;'>
        <span style='color:#00d2ff;'>ℹ️</span>
        <span style='color:#aaa;'> Paste any news article URL and FakeShield will
        automatically extract and analyze the content.</span>
    </div>""", unsafe_allow_html=True)

    url_input      = st.text_input("🔗 News Article URL", placeholder="https://www.example.com/news/article...")
    analyze_url_btn = st.button("🔍 FETCH & ANALYZE", key="analyze_url")

    if analyze_url_btn:
        if not url_input.startswith("http"):
            st.warning("⚠️ Please enter a valid URL starting with http/https")
        else:
            with st.spinner("🌐 Fetching article from URL..."):
                try:
                    import requests as req
                    from bs4 import BeautifulSoup
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                    response = req.get(url_input, headers=headers, timeout=10)
                    soup     = BeautifulSoup(response.text, 'lxml')
                    for tag in soup(['script','style','nav','footer','header','aside']):
                        tag.decompose()
                    fetched_title = ''
                    if soup.find('h1'):
                        fetched_title = soup.find('h1').get_text(strip=True)
                    elif soup.find('title'):
                        fetched_title = soup.find('title').get_text(strip=True)
                    paragraphs   = soup.find_all('p')
                    fetched_text = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
                    full_text    = f"{fetched_title} {fetched_text}".strip()

                    if len(full_text.split()) < 10:
                        st.error("❌ Could not extract enough text from URL.")
                    else:
                        st.success(f"✅ Article fetched: **{fetched_title[:80]}**")
                        with st.spinner("🤖 Analyzing..."):
                            result = predict_news(full_text, url=url_input)
                        st.markdown("---")
                        with st.expander("📄 Extracted Article Content"):
                            st.write(fetched_text[:1000] + "...")

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if result['label'] == 'FAKE':
                                st.markdown("<div class='result-fake'><div class='result-emoji'>🚨</div><div class='result-label'>FAKE NEWS</div></div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div class='result-real'><div class='result-emoji'>✅</div><div class='result-label'>REAL NEWS</div></div>", unsafe_allow_html=True)
                        with col2:
                            st.plotly_chart(render_gauge(result['confidence'], result['label']), use_container_width=True)

                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#00d2ff;'>{result['confidence']}%</div><div class='metric-label'>Confidence</div></div>", unsafe_allow_html=True)
                        with m2:
                            st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#00d2ff;'>{result['latency']}s</div><div class='metric-label'>Response Time</div></div>", unsafe_allow_html=True)
                        with m3:
                            st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#00d2ff;'>{len(full_text.split())}</div><div class='metric-label'>Words Analyzed</div></div>", unsafe_allow_html=True)
                        with m4:
                            st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{result['credibility_color']};font-size:1rem;'>{result['credibility_icon']} {result['credibility']}</div><div class='metric-label'>Source Check</div></div>",unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Failed to fetch article: {str(e)}")
                    st.info("💡 Try copying the article text and using the Text Analysis tab instead.")


# ════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Model Performance Comparison")
    import pandas as pd

    comparison_data = {
        'Model'    : ['Naive Bayes','Logistic Regression','BiLSTM','SVM','Hybrid v2\n(TF-IDF+BiLSTM)'],
        'Accuracy' : [88.15, 96.43, 96.81, 97.66, 97.84],
        'Precision': [87.51, 96.39, 95.99, 97.43, 97.83],
        'Recall'   : [90.19, 96.78, 97.99, 98.11, 97.99],
        'F1'       : [88.83, 96.59, 96.98, 97.77, 97.91],
    }
    df_comp = pd.DataFrame(comparison_data)

    fig_bar = go.Figure()
    for i, metric in enumerate(['Accuracy','Precision','Recall','F1']):
        fig_bar.add_trace(go.Bar(
            name=metric, x=df_comp['Model'], y=df_comp[metric],
            marker_color=['#7b2ff7','#00d2ff','#ff6b6b','#38ef7d'][i], opacity=0.85
        ))
    fig_bar.update_layout(
        barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', legend=dict(bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[85,100]),
        height=400, margin=dict(t=20,b=20),
        title=dict(text='All Models — Performance Metrics (%)', font=dict(color='white'))
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### 🕸️ Model Radar Chart")
    categories   = ['Accuracy','Precision','Recall','F1']
    fig_radar    = go.Figure()
    radar_colors = ['#aaa','#7b2ff7','#00d2ff','#ff6b6b','#38ef7d']
    for i, row in df_comp.iterrows():
        vals = [row['Accuracy'],row['Precision'],row['Recall'],row['F1']]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=categories+[categories[0]],
            name=row['Model'].replace('\n',' '),
            line=dict(color=radar_colors[i], width=2), fill='toself', opacity=0.3
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(range=[85,100], gridcolor='rgba(255,255,255,0.2)',
                           tickcolor='white', tickfont=dict(color='white',size=10)),
            angularaxis=dict(tickfont=dict(color='white',size=12), gridcolor='rgba(255,255,255,0.2)')
        ),
        paper_bgcolor='rgba(0,0,0,0)', font_color='white',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='white')), height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("#### 📋 Full Results Table")
    st.dataframe(
        df_comp.set_index('Model').style.background_gradient(cmap='plasma', axis=None).format("{:.2f}"),
        use_container_width=True
    )


# ════════════════════════════════════════════════════════════
# TAB 4 — LIVE NEWS
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📡 Live News Analysis")
    st.markdown("""
    <div style='background:rgba(0,210,255,0.08);border:1px solid rgba(0,210,255,0.2);
                border-radius:12px;padding:1rem;margin-bottom:1.5rem;'>
        <span style='color:#00d2ff;'>📡</span>
        <span style='color:#aaa;'> Fetch and analyze today's live news headlines in real-time.</span>
    </div>""", unsafe_allow_html=True)

    col_cat, col_src, col_num = st.columns([2, 2, 1])
    with col_cat:
        from config.config import NEWS_CATEGORIES
        category = st.selectbox("📂 Category", options=NEWS_CATEGORIES, index=0)
    with col_src:
        source_type = st.selectbox("📡 Source", options=["NewsAPI (Live)", "RSS Feeds (Backup)"])
    with col_num:
        num_articles = st.selectbox("📰 Count", [5, 10, 15], index=1)

    if source_type == "RSS Feeds (Backup)":
        from realtime.rss_fetcher import get_available_feeds
        rss_source = st.selectbox("Select RSS Feed", options=get_available_feeds())

    fetch_btn = st.button("📡 FETCH LIVE NEWS", key="fetch_live")

    if fetch_btn:
        with st.spinner("📡 Fetching live news..."):
            if source_type == "NewsAPI (Live)":
                from realtime.news_api import fetch_live_news
                articles, error = fetch_live_news(category=category, page_size=num_articles)
            else:
                from realtime.rss_fetcher import fetch_rss_news
                articles, error = fetch_rss_news(feed_name=rss_source, limit=num_articles)

        if error:
            st.error(f"❌ {error}")
            if "apiKey" in str(error) or "401" in str(error):
                st.info("💡 Check your API key in config/config.py")
        elif not articles:
            st.warning("No articles found. Try a different category.")
        else:
            st.success(f"✅ Fetched {len(articles)} articles!")
            st.markdown("---")

            fake_count   = 0
            real_count   = 0
            results_list = []
            progress     = st.progress(0)

            for i, article in enumerate(articles):
                text   = f"{article['title']} {article['description']}"
                result = predict_news(text, url=article['url'])
                results_list.append({
                    **article, **result,
                    'credibility': get_source_credibility(article['url'], article['source'])
                })
                if result['label'] == 'FAKE':
                    fake_count += 1
                else:
                    real_count += 1
                progress.progress((i + 1) / len(articles))
            progress.empty()

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#00d2ff;'>{len(articles)}</div><div class='metric-label'>Total Articles</div></div>", unsafe_allow_html=True)
            with s2:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#ff416c;'>{fake_count}</div><div class='metric-label'>Flagged Fake</div></div>", unsafe_allow_html=True)
            with s3:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#38ef7d;'>{real_count}</div><div class='metric-label'>Verified Real</div></div>", unsafe_allow_html=True)
            with s4:
                fake_pct = round(fake_count / len(articles) * 100)
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#ffd700;'>{fake_pct}%</div><div class='metric-label'>Fake Rate</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📰 Article Analysis")

            for i, r in enumerate(results_list):
                is_fake     = r['label'] == 'FAKE'
                card_color  = 'rgba(255,65,108,0.08)' if is_fake else 'rgba(56,239,125,0.08)'
                border_col  = '#ff416c' if is_fake else '#38ef7d'
                label_col   = '#ff416c' if is_fake else '#38ef7d'
                label_icon  = '🚨' if is_fake else '✅'
                title_safe  = html_lib.escape(str(r.get('title',''))[:100])
                desc_safe   = html_lib.escape(str(r.get('description',''))[:150])
                source_safe = html_lib.escape(str(r.get('source','Unknown')))
                url_safe    = html_lib.escape(str(r.get('url','#')))
                cred        = r.get('credibility', {})
                cred_label  = html_lib.escape(str(cred.get('label','Unknown')))
                cred_color  = cred.get('color','#888')
                cred_icon   = cred.get('icon','❓')
                confidence  = r.get('confidence', 0)

                card = (
                    f'<div style="background:{card_color};border:1px solid {border_col};'
                    f'border-radius:12px;padding:1.2rem;margin-bottom:1rem;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">'
                    f'<div style="flex:1;min-width:200px;">'
                    f'<div style="color:white;font-weight:700;font-size:0.95rem;line-height:1.4;margin-bottom:6px;">{title_safe}</div>'
                    f'<div style="color:#888;font-size:0.8rem;">📰 {source_safe} &nbsp;|&nbsp; '
                    f'{cred_icon} <span style="color:{cred_color};">{cred_label}</span></div>'
                    f'</div>'
                    f'<div style="text-align:right;">'
                    f'<div style="color:{label_col};font-weight:900;font-size:1.1rem;">{label_icon} {r["label"]}</div>'
                    f'<div style="color:#888;font-size:0.8rem;">{confidence}% confidence</div>'
                    f'</div></div>'
                    f'<div style="color:#aaa;font-size:0.82rem;margin-top:8px;line-height:1.5;">{desc_safe}...</div>'
                    f'<div style="margin-top:8px;">'
                    f'<a href="{url_safe}" target="_blank" style="color:#00d2ff;font-size:0.8rem;text-decoration:none;">'
                    f'🔗 Read Full Article</a></div></div>'
                )
                st.markdown(card, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — LLM COMPARISON
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🤖 LLM Comparison Analysis")
    st.markdown("""
<div style="background:rgba(123,47,247,0.08);border:1px solid rgba(123,47,247,0.3);
border-radius:12px;padding:1rem;margin-bottom:1.5rem;">
<span style="color:#7b2ff7;font-weight:700;">🔬 Research Feature</span>
<span style="color:#aaa;"> Compare your trained Hybrid Model against Google phi3 LLM
for the same article. This is the core research contribution of this project.</span>
</div>""", unsafe_allow_html=True)

    llm_headline = st.text_input("📰 News Headline", placeholder="Enter headline to compare both models...", key="llm_headline")
    llm_content  = st.text_area("📄 Article Content (optional)", placeholder="Paste article content for deeper analysis...", height=150, key="llm_content")
    compare_btn  = st.button("🔬 COMPARE MODELS", key="compare_llm")

    if compare_btn:
        if len(llm_headline.split()) < 3:
            st.warning("⚠️ Please enter at least a headline.")
        else:
            col_m, col_g = st.columns(2)
            with col_m:
                with st.spinner("🧠 Running Hybrid Model..."):
                    full_text = f"{llm_headline} {llm_content}".strip()
                    my_result = predict_news(full_text)
            with col_g:
                with st.spinner("🤖 Querying phi3..."):
                    from realtime.llm_analyzer import analyze_with_phi3, get_agreement_analysis
                    gem_result = analyze_with_phi3(llm_headline, llm_content)

            st.markdown("---")
            st.markdown("### 📊 Side-by-Side Comparison")
            col1, col2 = st.columns(2)

            my_color  = '#ff416c' if my_result['label']  == 'FAKE' else '#38ef7d'
            gem_color = '#ff416c' if gem_result['label'] == 'FAKE' else '#38ef7d'
            my_icon   = '🚨' if my_result['label']  == 'FAKE' else '✅'
            gem_icon  = '🚨' if gem_result['label'] == 'FAKE' else '✅'

            with col1:
                st.markdown(f"""
<div style="background:rgba(255,255,255,0.05);border:2px solid {my_color};
border-radius:16px;padding:1.5rem;text-align:center;">
<div style="font-size:0.85rem;color:#aaa;margin-bottom:8px;letter-spacing:2px;">YOUR HYBRID MODEL</div>
<div style="font-size:2.5rem;">{my_icon}</div>
<div style="font-size:2rem;font-weight:900;color:{my_color};letter-spacing:2px;">{my_result['label']}</div>
<div style="color:#aaa;font-size:0.9rem;margin-top:8px;">{my_result['confidence']}% confidence</div>
<div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:0.8rem;margin-top:12px;">
<div style="color:#888;font-size:0.75rem;">TF-IDF + BiLSTM</div>
<div style="color:#aaa;font-size:0.8rem;margin-top:4px;">Trained on 72K WELFake articles</div>
</div></div>""", unsafe_allow_html=True)

            with col2:
                if gem_result.get('error'):
                    st.error(f"❌ phi3 error: {gem_result['error']}")
                else:
                    st.markdown(f"""
<div style="background:rgba(255,255,255,0.05);border:2px solid {gem_color};
border-radius:16px;padding:1.5rem;text-align:center;">
<div style="font-size:0.85rem;color:#aaa;margin-bottom:8px;letter-spacing:2px;">LOCAL PHI-3 LLM</div>
<div style="font-size:2.5rem;">{gem_icon}</div>
<div style="font-size:2rem;font-weight:900;color:{gem_color};letter-spacing:2px;">{gem_result['label']}</div>
<div style="color:#aaa;font-size:0.9rem;margin-top:8px;">{gem_result['confidence']}% confidence</div>
<div style="background:rgba(255,255,255,0.05);border-radius:8px;padding:0.8rem;margin-top:12px;">
<div style="color:#888;font-size:0.75rem;">phi3 2.0 Flash Lite</div>
<div style="color:#aaa;font-size:0.8rem;margin-top:4px;">Real-time contextual understanding</div>
</div></div>""", unsafe_allow_html=True)

            if not gem_result.get('error'):
                agreement = get_agreement_analysis(
                    my_result['label'], gem_result['label'],
                    my_result['confidence'], gem_result['confidence']
                )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
<div style="background:rgba(255,255,255,0.05);border:1px solid {agreement['color']};
border-radius:12px;padding:1.2rem;text-align:center;">
<div style="font-size:1.5rem;">{agreement['icon']}</div>
<div style="color:{agreement['color']};font-weight:700;font-size:1.1rem;">{agreement['message']}</div>
<div style="color:#888;font-size:0.85rem;margin-top:4px;">
Reliability: <span style="color:{agreement['color']};">{agreement['reliability']}</span>
&nbsp;|&nbsp; Average Confidence: {agreement['avg_conf']}%
</div></div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 🧠 phi3's Analysis")
                st.markdown(f"""
<div style="background:rgba(123,47,247,0.08);border:1px solid rgba(123,47,247,0.3);
border-radius:12px;padding:1.2rem;">
<div style="color:white;font-size:0.95rem;line-height:1.7;">{gem_result['explanation']}</div>
</div>""", unsafe_allow_html=True)

                if gem_result.get('red_flags'):
                    st.markdown("#### 🚩 Red Flags Detected")
                    flags_html = ''.join([
                        f'<span style="background:rgba(255,65,108,0.15);border:1px solid #ff416c;'
                        f'border-radius:20px;padding:4px 12px;margin:4px;display:inline-block;'
                        f'color:#ff416c;font-size:0.85rem;">🚩 {flag}</span>'
                        for flag in gem_result['red_flags']
                    ])
                    st.markdown(f'<div style="margin-top:8px;">{flags_html}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if agreement['status'] == 'AGREE':
                    insight_color = '#38ef7d'
                    insight_text  = (
                        f"Both models independently classified this as <b>{my_result['label']}</b> "
                        f"with an average confidence of {agreement['avg_conf']}%. "
                        f"High agreement between traditional ML and LLM suggests a reliable prediction."
                    )
                else:
                    insight_color = '#ffd700'
                    insight_text  = (
                        f"Your Hybrid Model predicts <b>{my_result['label']}</b> ({my_result['confidence']}%) "
                        f"while phi3 predicts <b>{gem_result['label']}</b> ({gem_result['confidence']}%). "
                        f"Disagreement may indicate temporal distribution shift — this article may be "
                        f"outside the model's training distribution."
                    )

                st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border-left:3px solid {insight_color};
border-radius:0 8px 8px 0;padding:1rem;margin-top:8px;">
<div style="color:{insight_color};font-weight:700;font-size:0.85rem;margin-bottom:4px;">
📋 RESEARCH INSIGHT</div>
<div style="color:#aaa;font-size:0.85rem;line-height:1.7;">{insight_text}</div>
</div>""", unsafe_allow_html=True)