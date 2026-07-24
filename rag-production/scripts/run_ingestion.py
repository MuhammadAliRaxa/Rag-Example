"""CLI script to trigger ingestion pipeline."""
import argparse
from src.ingestion.pipeline import IngestionPipeline

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system.")
    parser.add_argument("--path", type=str, required=True, help="Path to file or directory to ingest")
    args = parser.parse_args()

    pipeline = IngestionPipeline()
    res = pipeline.process_file(args.path)
    print(f"Ingestion finished: {res}")

if __name__ == "__main__":
    main()
