"""
database/db.py — SQLite logging for all FakeShield predictions
"""

import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH  = os.path.join(BASE_DIR, 'database', 'predictions.db')


def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp          TEXT    NOT NULL,
            mode               TEXT    NOT NULL,
            input_text         TEXT,
            source_url         TEXT,
            source_credibility TEXT,
            hybrid_label       TEXT,
            hybrid_confidence  REAL,
            svm_label          TEXT,
            models_agree       INTEGER,
            latency            REAL,
            llm_label          TEXT,
            llm_confidence     REAL,
            llm_agreement      TEXT
        )
    ''')
    conn.commit()
    conn.close()


def log_prediction(
    mode,
    input_text        = '',
    source_url        = '',
    source_credibility= '',
    hybrid_label      = '',
    hybrid_confidence = 0.0,
    svm_label         = '',
    models_agree      = False,
    latency           = 0.0,
    llm_label         = None,
    llm_confidence    = None,
    llm_agreement     = None
):
    """Insert one prediction row."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute('''
        INSERT INTO predictions (
            timestamp, mode, input_text, source_url,
            source_credibility, hybrid_label, hybrid_confidence,
            svm_label, models_agree, latency,
            llm_label, llm_confidence, llm_agreement
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        mode,
        input_text[:300] if input_text else '',
        source_url,
        source_credibility,
        hybrid_label,
        hybrid_confidence,
        svm_label,
        1 if models_agree else 0,
        latency,
        llm_label,
        llm_confidence,
        llm_agreement
    ))
    conn.commit()
    conn.close()


def get_recent_predictions(limit=50):
    """Return the most recent predictions as a list of dicts."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c    = conn.cursor()
    c.execute(
        'SELECT * FROM predictions ORDER BY id DESC LIMIT ?',
        (limit,)
    )
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows


def get_statistics():
    """Return summary stats as a dict."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    stats = {}

    c.execute('SELECT COUNT(*) FROM predictions')
    stats['total'] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM predictions WHERE hybrid_label='FAKE'")
    stats['fake_count'] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM predictions WHERE hybrid_label='REAL'")
    stats['real_count'] = c.fetchone()[0]

    c.execute('SELECT AVG(hybrid_confidence) FROM predictions')
    avg = c.fetchone()[0]
    stats['avg_confidence'] = round(avg, 2) if avg else 0.0

    c.execute('SELECT AVG(latency) FROM predictions')
    avg_lat = c.fetchone()[0]
    stats['avg_latency'] = round(avg_lat, 3) if avg_lat else 0.0

    c.execute('SELECT COUNT(*) FROM predictions WHERE models_agree=1')
    stats['agree_count'] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM predictions WHERE llm_label IS NOT NULL")
    stats['llm_comparisons'] = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM predictions WHERE llm_agreement='AGREE'")
    stats['llm_agree_count'] = c.fetchone()[0]

    stats['fake_rate'] = (
        round(stats['fake_count'] / stats['total'] * 100, 1)
        if stats['total'] > 0 else 0
    )

    conn.close()
    return stats


def clear_history():
    """Delete all prediction rows."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.cursor().execute('DELETE FROM predictions')
    conn.commit()
    conn.close()