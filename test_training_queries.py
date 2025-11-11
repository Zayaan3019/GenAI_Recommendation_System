"""
FIXED: Test system with all training queries
"""
from backend.app.utils.evaluation import RetrievalEvaluator
from backend.app.utils.embeddings_local import LocalEmbeddingGenerator
from backend.app.utils.retrieval_optimized import OptimizedHybridRetriever
from backend.app.utils.balance import RecommendationBalancer
from backend.app.config import get_settings
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    settings = get_settings()
    
    # Load system
    with open(settings.ASSESSMENTS_PATH, 'r') as f:
        data = json.load(f)
    
    assessments = data['assessments']
    embeddings = np.array(data['embeddings'], dtype=np.float32)
    
    # Get model from metadata (IMPORTANT!)
    model_name = data['metadata'].get('model', 'all-mpnet-base-v2')
    
    logger.info(f"Using model: {model_name}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Initialize with SAME model
    embedding_gen = LocalEmbeddingGenerator(model_name=model_name)  # FIXED
    retriever = OptimizedHybridRetriever(assessments, embeddings, alpha=0.3)
    balancer = RecommendationBalancer()
    
    # Load evaluator
    evaluator = RetrievalEvaluator(r"C:/Users/Mohamed Zayaan/Downloads/Gen_AI Dataset.xlsx")
    
    # Test each training query
    logger.info("\n" + "="*80)
    logger.info("TESTING TRAINING QUERIES")
    logger.info("="*80)
    
    for idx, query in enumerate(evaluator.ground_truth.keys(), 1):
        logger.info(f"\n[Query {idx}/10] {query[:80]}...")
        
        # Get recommendations
        query_embedding = embedding_gen.generate_query_embedding(query)
        recommendations = retriever.retrieve(query, query_embedding, top_k=10, use_reranking=True)
        balanced = balancer.balance_recommendations(recommendations, target_count=10)
        
        # Show top 3 results
        logger.info("  Top 3 recommendations:")
        for i, rec in enumerate(balanced[:3], 1):
            logger.info(f"    {i}. {rec['assessment_name']} (score: {rec['score']:.4f})")
        
        # Calculate recall
        recommended_urls = [rec['url'] for rec in balanced]
        recall = evaluator.recall_at_k(query, recommended_urls, k=10)
        logger.info(f"  Recall@10: {recall:.4f}")
    
    # Overall evaluation
    def retriever_func(query, top_k=10):
        query_embedding = embedding_gen.generate_query_embedding(query)
        recs = retriever.retrieve(query, query_embedding, top_k=top_k, use_reranking=True)
        balanced = balancer.balance_recommendations(recs, target_count=top_k)
        return [rec['url'] for rec in balanced]
    
    mean_recall, details = evaluator.evaluate_on_train(retriever_func, k=10)
    
    logger.info("\n" + "="*80)
    logger.info(f"FINAL MEAN RECALL@10: {mean_recall:.4f}")
    logger.info("="*80)

if __name__ == "__main__":
    main()

