# InterrogationX

Forensic statement analysis platform that detects contradictions and similarities between defendant and witness statements using NLP.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Web_App-green)
![SpaCy](https://img.shields.io/badge/SpaCy-NLP-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## What It Does

InterrogationX compares a suspect's statement against one or more witness statements and classifies sentence pairs as contradictions, corroborations, or partial matches. It uses 8 detection channels:

- **Negation analysis** — detects opposing verb polarity ("I stopped" vs "I didn't stop")
- **Antonym detection** — catches opposite words ("fast" vs "slow", "yellow" vs "red")
- **Verb conflict** — flags contradicting actions ("leave" vs "stay", "grab" vs "release")
- **Numerical discrepancy** — spots different counts or quantities
- **Temporal conflict** — identifies mismatched times or time-of-day references
- **Entity mismatch** — detects different people or locations in related sentences
- **Quantifier opposition** — catches "all" vs "none", "always" vs "never"
- **Property mismatch** — finds conflicting descriptions of the same subject

Scores are confidence-weighted, not simple ratios. Sentences must be about the same topic (measured by embedding similarity) to qualify as contradictions.

## Tech Stack

Python, Flask, SpaCy, Sentence-Transformers, Scikit-learn, Pandas, LangChain, ChromaDB, SQLite

## Quick Start

### Local

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
```

Open http://localhost:5000

### Docker

```bash
docker compose up --build
```

## Project Structure

```
app.py              Flask routes and entry point
analyzer.py         StatementAnalyzer (8 detection channels)
constants.py        Linguistic data and thresholds
database.py         SQLite persistence
templates/
  index.html        Frontend HTML
static/
  css/style.css     Styles
  js/app.js         Client-side logic
forensic_statements_data_en.json   20 sample forensic cases
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/cases` | List all cases |
| GET | `/api/case/<id>` | Get a single case |
| POST | `/api/analyze` | Analyze a dataset case (`{"case_id": "001"}`) |
| POST | `/api/analyze/custom` | Analyze custom statements |
| GET | `/api/history` | Get past analyses |

## Verdicts

| Verdict | Meaning |
|---------|---------|
| HIGHLY CONTRADICTORY | Strong contradiction signals detected |
| CONTRADICTORY | Notable contradictions found |
| INCONCLUSIVE | No clear pattern |
| CORROBORATING | Statements mostly align |
| STRONGLY CORROBORATING | High agreement across statements |

## License

MIT
