#!/usr/bin/env bash
set -e
echo "Running RAG Evaluation Suite..."
python -m src.evaluation.run_eval
