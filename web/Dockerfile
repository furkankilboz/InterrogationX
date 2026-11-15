# InterrogationX – Container Build
# Build:   docker build -t interrogationx .
# Run:     docker run -p 5000:5000 interrogationx
# Compose: docker compose up

FROM python:3.11-slim

LABEL maintainer="InterrogationX"
LABEL description="Forensic Statement Contradiction & Similarity Analyzer"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm


RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('paraphrase-MiniLM-L6-v2')"

COPY . .

VOLUME ["/app/chroma_store"]

EXPOSE 5000

ENV FLASK_APP=app.py \
    FLASK_ENV=production \
    IXD_DATA=/app/forensic_statements_data_en.json \
    IXD_DB=/app/interrogation_x.db \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:5000/api/cases || exit 1

CMD ["python", "app.py"]
