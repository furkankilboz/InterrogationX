import json

from flask import Flask, request, jsonify, render_template

from constants import DATA_FILE
from database import (
    setup_database,
    import_json_cases,
    save_analysis_result,
    get_history,
    get_pairs_for_analysis,
    json_safe,
)
from analyzer import StatementAnalyzer

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = StatementAnalyzer()
    return _analyzer


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cases")
def api_cases():
    try:
        with open(DATA_FILE, encoding="utf-8") as fh:
            cases = json.load(fh)
        return jsonify({
            "cases": [
                {
                    "case_id": c["case_id"],
                    "description": c["description"],
                    "witness_count": len(c.get("witness_statements", [])),
                }
                for c in cases
            ],
            "total": len(cases),
        })
    except FileNotFoundError:
        return jsonify({"error": "Data file not found"}), 404


@app.route("/api/case/<case_id>")
def api_case(case_id):
    try:
        with open(DATA_FILE, encoding="utf-8") as fh:
            cases = json.load(fh)
    except FileNotFoundError:
        return jsonify({"error": "Data file not found"}), 404
    case = next((c for c in cases if c["case_id"] == case_id), None)
    if not case:
        return jsonify({"error": f"Case {case_id} not found"}), 404
    return jsonify(case)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    body = request.get_json(force=True, silent=True) or {}
    case_id = body.get("case_id", "").strip()
    if not case_id:
        return jsonify({"error": "case_id is required"}), 400
    try:
        with open(DATA_FILE, encoding="utf-8") as fh:
            cases = json.load(fh)
    except FileNotFoundError:
        return jsonify({"error": "Data file missing"}), 500
    case = next((c for c in cases if c["case_id"] == case_id), None)
    if not case:
        return jsonify({"error": f"Case {case_id} not found"}), 404

    result = _get_analyzer().analyze(
        case["suspect_statement"],
        case["witness_statements"],
        case_id,
    )
    try:
        result["analysis_id"] = save_analysis_result(case_id, result)
    except Exception as exc:
        app.logger.warning("DB save failed: %s", exc)
    return jsonify(json_safe(result))


@app.route("/api/analyze/custom", methods=["POST"])
def api_analyze_custom():
    body = request.get_json(force=True, silent=True) or {}
    suspect = body.get("suspect_statement", "").strip()
    witnesses = body.get("witness_statements", [])
    if not suspect:
        return jsonify({"error": "suspect_statement is required"}), 400
    if not witnesses or not isinstance(witnesses, list):
        return jsonify({"error": "witness_statements (list) is required"}), 400
    case_id = body.get("case_id", "custom")
    result = _get_analyzer().analyze(suspect, witnesses, case_id)
    try:
        result["analysis_id"] = save_analysis_result(case_id, result)
    except Exception:
        pass
    return jsonify(json_safe(result))


@app.route("/api/history")
def api_history():
    rows = get_history()
    return jsonify({"analyses": rows, "total": len(rows)})


@app.route("/api/history/<int:aid>/pairs")
def api_history_pairs(aid):
    pairs = get_pairs_for_analysis(aid)
    return jsonify({"analysis_id": aid, "pairs": pairs, "total": len(pairs)})


if __name__ == "__main__":
    setup_database()
    import_json_cases()
    print("\n  InterrogationX running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
