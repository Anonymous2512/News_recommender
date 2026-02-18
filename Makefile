# Shortcuts. On Windows without make, run the commands shown after each target directly.
PY ?= python
MIND ?= MINDsmall_train

.PHONY: install install-embed test integration lint typecheck demo prepare embed train evaluate import api compose-demo compose-up benchmark dashboard

install:            ## serving + dev dependencies (no torch)
	$(PY) -m pip install -e ".[serve,dev]"
install-embed:      ## MiniLM encoder for the offline embedding job
	$(PY) -m pip install -e ".[embed]"
test:
	$(PY) -m pytest
integration:        ## needs Redis: docker compose up -d redis
	REDIS_URL=redis://localhost:6379/15 $(PY) -m pytest -m integration
lint:
	$(PY) -m ruff check newsrec app tests optimizer.py ranker.py embeddings.py mind_preprocessing.py
typecheck:
	$(PY) -m mypy
demo:               ## SYNTHETIC end-to-end demo (hashing encoder)
	$(PY) -m newsrec demo
prepare:
	$(PY) -m newsrec prepare --news $(MIND)/news.tsv --behaviors $(MIND)/behaviors.tsv --out data/processed
embed:
	$(PY) -m newsrec embed --processed data/processed --backend minilm
train:
	$(PY) -m newsrec train --processed data/processed --artifacts artifacts
evaluate:
	$(PY) -m newsrec evaluate --artifact-dir artifacts/$$(cat artifacts/CURRENT) --processed data/processed
import:
	$(PY) -m newsrec import-catalog --artifacts artifacts --database-url sqlite:///data/newsrec.db
api:
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000
compose-demo:
	docker compose --profile demo run --rm bootstrap
compose-up:
	docker compose up -d --build redis api
benchmark:
	$(PY) scripts/benchmark.py --requests 200
dashboard:
	$(PY) -m streamlit run dashboard.py
