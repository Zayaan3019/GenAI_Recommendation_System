"""
FIXED: Find optimal alpha value with correct model
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from backend.app.config import get_settings
from backend.app.utils.embeddings_local import LocalEmbeddingGenerator
from backend.app.utils.retrieval_optimized import OptimizedHybridRetriever
from backend.app.utils.evaluation import RetrievalEvaluator
import json
import numpy as np

def test_alpha(alpha_value):
    settings = get_settings()
    
    # Load data
    with open(settings.ASSESSMENTS_PATH, 'r') as f:
        data = json.load(f)
    
    assessments = data['assessments']
    embeddings = np.array(data['embeddings'], dtype=np.float32)
    
    # Get model from metadata (IMPORTANT!)
    model_name = data['metadata'].get('model', 'all-mpnet-base-v2')
    
    print(f"  Using model: {model_name}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    # Initialize with SAME model used for embeddings
    emb_gen = LocalEmbeddingGenerator(model_name=model_name)  # FIXED
    retriever = OptimizedHybridRetriever(assessments, embeddings, alpha=alpha_value)
    evaluator = RetrievalEvaluator(r"C:/Users/Mohamed Zayaan/Downloads/Gen_AI Dataset.xlsx")
    
    # Test
    def retriever_func(query, top_k=10):
        query_emb = emb_gen.generate_query_embedding(query)
        recs = retriever.retrieve(query, query_emb, top_k=top_k, use_reranking=True)
        return [rec['url'] for rec in recs]
    
    recall, _ = evaluator.evaluate_on_train(retriever_func, k=10)
    return recall

print("\n" + "="*80)
print("ALPHA TUNING - Finding optimal alpha value")
print("="*80)

# Test first to see model info
print("\nChecking configuration...")
settings = get_settings()
with open(settings.ASSESSMENTS_PATH, 'r') as f:
    data = json.load(f)
model_name = data['metadata'].get('model', 'unknown')
emb_dim = data['metadata'].get('dim', 'unknown')

print(f"✓ Loaded model: {model_name}")
print(f"✓ Embedding dimension: {emb_dim}")
print(f"✓ Total assessments: {len(data['assessments'])}")

print("\n" + "="*80)
print("Testing different alpha values...")
print("="*80)

alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
results = []

for alpha in alphas:
    print(f"\nTesting alpha = {alpha}...")
    recall = test_alpha(alpha)
    results.append((alpha, recall))
    print(f"  ✓ Recall@10: {recall:.4f}")

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
for alpha, recall in sorted(results, key=lambda x: x[1], reverse=True):
    bar = "█" * int(recall * 50)
    print(f"alpha={alpha:.1f}  →  Recall@10: {recall:.4f}  {bar}")

best_alpha, best_recall = max(results, key=lambda x: x[1])
print("\n" + "="*80)
print(f"✓ BEST: alpha={best_alpha} with Recall@10={best_recall:.4f}")
print("="*80)
print(f"\nUpdate this value in:")
print(f"  - backend/app/main.py (line with 'alpha=')")
print(f"  - backend/app/utils/retrieval_optimized.py (line with 'alpha=')")

