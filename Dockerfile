# API image: serving dependencies only. MiniLM/torch are NOT needed at serve time (article vectors come
# from the artifact); build embeddings offline with `pip install .[embed]` + `python -m newsrec embed`.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY pyproject.toml README.md ./
COPY newsrec ./newsrec
COPY app ./app
COPY optimizer.py ranker.py mind_preprocessing.py embeddings.py ./
RUN pip install ".[serve]" && useradd --create-home --uid 10001 newsrec && mkdir -p /srv/data /srv/artifacts \
    && chown -R newsrec /srv
USER newsrec

ENV NEWSREC_ARTIFACTS_DIR=/srv/artifacts NEWSREC_DATABASE_URL=sqlite:////srv/data/newsrec.db
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
