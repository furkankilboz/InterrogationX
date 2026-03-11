import json
import math
import os
import sqlite3

import numpy as np
import pandas as pd

from constants import DB_FILE, DATA_FILE


def _db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def setup_database():
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cases (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id     TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS analyses (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id             TEXT NOT NULL,
                overall_verdict     TEXT,
                contradiction_score REAL,
                corroboration_score REAL,
                full_json           TEXT,
                created_at          TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS sentence_pairs (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id      INTEGER NOT NULL,
                witness_idx      INTEGER,
                suspect_sentence TEXT,
                witness_sentence TEXT,
                relationship     TEXT,
                similarity       REAL,
                confidence       REAL,
                reasons          TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id)
            );
            CREATE INDEX IF NOT EXISTS idx_analyses_case
                ON analyses(case_id);
            CREATE INDEX IF NOT EXISTS idx_pairs_analysis
                ON sentence_pairs(analysis_id);
        """)


def import_json_cases():
    if not os.path.exists(DATA_FILE):
        return
    with open(DATA_FILE, encoding="utf-8") as fh:
        cases = json.load(fh)
    with _db() as conn:
        for c in cases:
            conn.execute(
                "INSERT OR IGNORE INTO cases (case_id, description) VALUES (?, ?)",
                (c["case_id"], c["description"]),
            )


def save_analysis_result(case_id, result):
    safe = json_safe(result)
    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO analyses
                   (case_id, overall_verdict, contradiction_score,
                    corroboration_score, full_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                case_id,
                result.get("overall_verdict", ""),
                result.get("overall_contradiction_score", 0),
                result.get("overall_corroboration_score", 0),
                json.dumps(safe, ensure_ascii=False),
            ),
        )
        aid = cur.lastrowid
        for wa in result.get("witness_analyses", []):
            for p in wa.get("pairwise_analysis", []):
                conn.execute(
                    """INSERT INTO sentence_pairs
                           (analysis_id, witness_idx, suspect_sentence,
                            witness_sentence, relationship, similarity,
                            confidence, reasons)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        aid,
                        wa["witness_idx"],
                        p["suspect_sentence"],
                        p["witness_sentence"],
                        p["relationship"],
                        p["similarity"],
                        p.get("confidence", 0),
                        json.dumps(p.get("reasons", [])),
                    ),
                )
    return aid


def get_history():
    with _db() as conn:
        rows = conn.execute(
            """SELECT id, case_id, overall_verdict,
                      contradiction_score, corroboration_score, created_at
               FROM analyses ORDER BY created_at DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def get_pairs_for_analysis(analysis_id):
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM sentence_pairs WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["reasons"] = ", ".join(json.loads(d["reasons"])) if d["reasons"] else ""
        result.append(d)
    return result


def json_safe(obj):
    if isinstance(obj, dict):
        return {
            k: json_safe(v)
            for k, v in obj.items()
            if k not in ("summary_df",)
        }
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return obj
