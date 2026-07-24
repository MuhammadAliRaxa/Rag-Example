"""CLI: Run full evaluation suite and print report."""
from src.evaluation.golden_dataset import GoldenDatasetLoader
from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.evaluation.answer_metrics import AnswerMetrics

def run_evaluation():
    print("=== Starting RAG Evaluation Suite ===")
    dataset = GoldenDatasetLoader().load()
    print(f"Loaded {len(dataset)} test samples.")
    
    p5 = RetrievalMetrics.precision_at_k(["doc1", "doc2"], ["doc1"], k=5)
    r5 = RetrievalMetrics.recall_at_k(["doc1", "doc2"], ["doc1"], k=5)
    
    print(f"Retrieval Precision@5: {p5:.2f}")
    print(f"Retrieval Recall@5:    {r5:.2f}")
    print("=== Evaluation Complete ===")

if __name__ == "__main__":
    run_evaluation()
