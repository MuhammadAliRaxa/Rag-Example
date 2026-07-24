.PHONY: help serve ingest eval test clean

help:
	@echo "Available commands:"
	@echo "  make serve    - Start the FastAPI dev server"
	@echo "  make ingest   - Run document ingestion pipeline"
	@echo "  make eval     - Run evaluation suite"
	@echo "  make test     - Run pytest unit and integration tests"
	@echo "  make clean    - Clean python cache files"

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

ingest:
	python scripts/run_ingestion.py --path $(or $(FILE),data/raw)

eval:
	python src/evaluation/run_eval.py

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
