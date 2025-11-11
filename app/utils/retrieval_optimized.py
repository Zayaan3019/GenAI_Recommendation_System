"""
OPTIMIZED retrieval with better scoring
"""
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

class OptimizedHybridRetriever:
    """
    Optimized retrieval with better fusion and reranking
    """
    
    def __init__(self, assessments: List[Dict], embeddings: np.ndarray, 
                 alpha: float = 0.3):  # OPTIMIZED: More weight on semantic
        self.assessments = assessments
        self.embeddings = embeddings
        self.alpha = alpha  # Lower = more semantic weight
        
        self._init_bm25()
        
        # Load reranker
        logger.info("Loading cross-encoder...")
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker_available = True
            logger.info("✓ Reranker loaded")
        except Exception as e:
            logger.warning(f"Reranker failed: {e}")
            self.reranker_available = False
    
    def _init_bm25(self):
        """Initialize BM25 with optimized tokenization"""
        logger.info("Building BM25 index...")
        
        corpus = []
        for assessment in self.assessments:
            # More comprehensive text
            text = f"{assessment.get('assessment_name', '')} "
            text += f"{assessment.get('assessment_name', '')} "  # Repeat
            text += f"{assessment.get('description', '')} "
            text += f"{assessment.get('test_type', '')} "
            text += " ".join(assessment.get('skills', []))
            text += f" {assessment.get('full_content', '')}"
            
            # Better tokenization
            tokens = text.lower().split()
            corpus.append(tokens)
        
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"✓ BM25 index built ({len(corpus)} docs)")
    
    def _bm25_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """BM25 search with more candidates"""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]
    
    def _semantic_search(self, query_embedding: np.ndarray, top_k: int = 50) -> List[Tuple[int, float]]:
        """Semantic search with more candidates"""
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Optimized hybrid search with better normalization
        """
        bm25_results = self._bm25_search(query, top_k)
        semantic_results = self._semantic_search(query_embedding, top_k)
        
        # Min-max normalization
        def normalize(results):
            scores = np.array([score for _, score in results])
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                scores = np.ones_like(scores)
            return [(idx, score) for (idx, _), score in zip(results, scores)]
        
        bm25_norm = normalize(bm25_results)
        semantic_norm = normalize(semantic_results)
        
        # Combine with optimized alpha (more weight on semantic)
        combined = {}
        for idx, score in bm25_norm:
            combined[idx] = self.alpha * score
        
        for idx, score in semantic_norm:
            if idx in combined:
                combined[idx] += (1 - self.alpha) * score
            else:
                combined[idx] = (1 - self.alpha) * score
        
        # Sort and return top-k
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def rerank(self, query: str, candidate_indices: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Rerank with cross-encoder
        """
        if not self.reranker_available:
            return [(idx, 1.0) for idx in candidate_indices[:top_k]]
        
        # Prepare pairs
        pairs = []
        for idx in candidate_indices:
            doc = f"{self.assessments[idx].get('assessment_name', '')} "
            doc += f"{self.assessments[idx].get('description', '')[:300]}"
            pairs.append([query, doc])
        
        # Rerank
        logger.debug(f"Reranking {len(pairs)} candidates...")
        try:
            scores = self.reranker.predict(pairs)
            
            # Normalize scores with sigmoid
            normalized_scores = 1.0 / (1.0 + np.exp(-np.array(scores)))
            
            # Sort
            ranked = sorted(
                zip(candidate_indices, normalized_scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            return ranked[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return [(idx, 1.0) for idx in candidate_indices[:top_k]]
    
    def retrieve(self, query: str, query_embedding: np.ndarray, 
                 top_k: int = 10, use_reranking: bool = True) -> List[Dict]:
        """
        Main retrieval with optimization
        """
        # Hybrid search with more candidates
        hybrid_results = self.hybrid_search(query, query_embedding, top_k=50)
        candidate_indices = [idx for idx, _ in hybrid_results]
        
        # Rerank
        if use_reranking and self.reranker_available:
            reranked = self.rerank(query, candidate_indices, top_k=top_k)
            final_indices = [idx for idx, _ in reranked]
            final_scores = [score for _, score in reranked]
        else:
            final_indices = candidate_indices[:top_k]
            final_scores = [score for _, score in hybrid_results[:top_k]]
        
        # Prepare results
        results = []
        for idx, score in zip(final_indices, final_scores):
            assessment = self.assessments[idx].copy()
            assessment['score'] = float(max(0.0, min(1.0, score)))
            results.append(assessment)
        
        return results
