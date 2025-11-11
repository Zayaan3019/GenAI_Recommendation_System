"""
FIXED: Generate predictions for the test set
"""
import os
import sys
import json
import numpy as np
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.app.config import get_settings
from backend.app.utils.embeddings_local import LocalEmbeddingGenerator
from backend.app.utils.retrieval_optimized import OptimizedHybridRetriever
from backend.app.utils.balance import RecommendationBalancer
from backend.app.utils.evaluation import RetrievalEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_system_components(settings):
    """Load all system components with correct model."""
    logger.info("Loading system components...")
    
    # Load assessments data
    with open(settings.ASSESSMENTS_PATH, 'r') as f:
        data = json.load(f)
    
    assessments = data['assessments']
    embeddings = np.array(data['embeddings'], dtype=np.float32)
    
    # Get model from metadata (IMPORTANT!)
    model_name = data['metadata'].get('model', 'all-mpnet-base-v2')
    
    logger.info(f"✓ Loaded {len(assessments)} assessments")
    logger.info(f"✓ Embeddings shape: {embeddings.shape}")
    logger.info(f"✓ Model: {model_name}")
    
    # Initialize with SAME model
    embedding_gen = LocalEmbeddingGenerator(model_name=model_name)  # FIXED
    
    retriever = OptimizedHybridRetriever(
        assessments=assessments,
        embeddings=embeddings,
        alpha=0.3
    )
    
    balancer = RecommendationBalancer(
        min_recommendations=settings.MIN_RECOMMENDATIONS,
        max_recommendations=settings.MAX_RECOMMENDATIONS
    )
    
    return embedding_gen, retriever, balancer, assessments

def retriever_function(query, embedding_gen, retriever, balancer, top_k=10):
    """Wrapper function for retrieval."""
    query_embedding = embedding_gen.generate_query_embedding(query)
    recommendations = retriever.retrieve(query, query_embedding, top_k=top_k, use_reranking=True)
    balanced_recs = balancer.balance_recommendations(recommendations, target_count=top_k)
    return balanced_recs

def main():
    logger.info("="*80)
    logger.info("GENERATING TEST SET PREDICTIONS")
    logger.info("="*80)
    
    settings = get_settings()
    
    # Load components
    logger.info("\n[STEP 1/3] Loading system components...")
    try:
        embedding_gen, retriever, balancer, assessments = load_system_components(settings)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Please run 'python scripts/setup_complete.py' first!")
        return
    
    # Initialize evaluator
    logger.info("\n[STEP 2/3] Loading ground truth data...")
    evaluator = RetrievalEvaluator(r"C:/Users/Mohamed Zayaan/Downloads/Gen_AI Dataset.xlsx")
    
    # Evaluate on training set
    logger.info("\n[STEP 3/3] Evaluating on training set...")
    
    def train_retriever_func(query, top_k=10):
        recs = retriever_function(query, embedding_gen, retriever, balancer, top_k)
        return [rec['url'] for rec in recs]
    
    train_recall, train_details = evaluator.evaluate_on_train(train_retriever_func, k=10)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING SET RESULTS")
    logger.info("="*80)
    logger.info(f"Mean Recall@10: {train_recall:.4f}")
    
    # Generate test predictions
    logger.info("\nGenerating test set predictions...")
    
    def test_retriever_func(query, top_k=10):
        return retriever_function(query, embedding_gen, retriever, balancer, top_k)
    
    pred_df = evaluator.generate_test_predictions(
        test_retriever_func,
        k=10,
        output_path="predictions.csv"
    )
    
    logger.info("\n" + "="*80)
    logger.info("COMPLETE!")
    logger.info("="*80)
    logger.info(f"✓ Training Recall@10: {train_recall:.4f}")
    logger.info(f"✓ Test predictions: predictions.csv ({len(pred_df)} rows)")

if __name__ == "__main__":
    main()
