import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Advanced hybrid retrieval system combining BM25 (lexical) and 
    semantic search with cross-encoder reranking.
    """
    
    def __init__(self, assessments: List[Dict], embeddings: np.ndarray, 
                 alpha: float = 0.5, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            assessments: List of assessment documents
            embeddings: Pre-computed semantic embeddings
            alpha: Weight for combining BM25 and semantic scores (0-1)
            reranker_model: Cross-encoder model for reranking
        """
        self.assessments = assessments
        self.embeddings = embeddings
        self.alpha = alpha
        
        # Initialize BM25
        self._init_bm25()
        
        # Initialize cross-encoder for reranking
        logger.info(f"Loading cross-encoder: {reranker_model}")
        try:
            self.reranker = CrossEncoder(reranker_model)
            self.reranker_available = True
            logger.info("✓ Cross-encoder loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
            self.reranker_available = False
    
    def _init_bm25(self):
        """Initialize BM25 index."""
        logger.info("Building BM25 index...")
        
        # Create searchable text corpus
        corpus = []
        for assessment in self.assessments:
            # Combine all relevant fields
            text = f"{assessment.get('assessment_name', '')} "
            text += f"{assessment.get('description', '')} "
            text += f"{assessment.get('test_type', '')} "
            text += " ".join(assessment.get('skills', []))
            text += f" {assessment.get('full_content', '')}"
            
            corpus.append(text.lower().split())
        
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built with {len(corpus)} documents")
    
    def _bm25_search(self, query: str, top_k: int = 30) -> List[Tuple[int, float]]:
        """Perform BM25 lexical search."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(idx, scores[idx]) for idx in top_indices]
        
        return results
    
    def _semantic_search(self, query_embedding: np.ndarray, top_k: int = 30) -> List[Tuple[int, float]]:
        """Perform semantic similarity search."""
        # Compute cosine similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        )[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices]
        
        return results
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int = 30) -> List[Tuple[int, float]]:
        """
        Combine BM25 and semantic search using weighted fusion.
        """
        # Get results from both methods
        bm25_results = self._bm25_search(query, top_k)
        semantic_results = self._semantic_search(query_embedding, top_k)
        
        # Normalize scores to [0, 1]
        def normalize_scores(results):
            scores = np.array([score for _, score in results])
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                scores = np.ones_like(scores)
            return [(idx, score) for (idx, _), score in zip(results, scores)]
        
        bm25_norm = normalize_scores(bm25_results)
        semantic_norm = normalize_scores(semantic_results)
        
        # Combine scores
        combined_scores = {}
        for idx, score in bm25_norm:
            combined_scores[idx] = self.alpha * score
        
        for idx, score in semantic_norm:
            if idx in combined_scores:
                combined_scores[idx] += (1 - self.alpha) * score
            else:
                combined_scores[idx] = (1 - self.alpha) * score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results[:top_k]
    
    def _normalize_reranker_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize cross-encoder scores to [0, 1] range.
        
        Cross-encoder scores are unbounded, so we use sigmoid normalization
        to map them to a probability-like range.
        """
        # Apply sigmoid to normalize to [0, 1]
        # sigmoid(x) = 1 / (1 + exp(-x))
        normalized = 1.0 / (1.0 + np.exp(-scores))
        
        return normalized
    
    def rerank(self, query: str, candidate_indices: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Rerank candidates using cross-encoder for higher precision.
        """
        if not self.reranker_available:
            # If reranker not available, return candidates with their scores
            logger.warning("Reranker not available, returning candidates in order")
            return [(idx, 1.0) for idx in candidate_indices[:top_k]]
        
        # Prepare query-document pairs
        pairs = []
        for idx in candidate_indices:
            doc_text = f"{self.assessments[idx].get('assessment_name', '')}. "
            doc_text += f"{self.assessments[idx].get('description', '')}"
            pairs.append([query, doc_text])
        
        # Get reranking scores
        logger.info(f"Reranking {len(pairs)} candidates...")
        try:
            raw_scores = self.reranker.predict(pairs)
            
            # Normalize scores to [0, 1]
            normalized_scores = self._normalize_reranker_scores(raw_scores)
            
            # Sort by score
            ranked = sorted(zip(candidate_indices, normalized_scores), key=lambda x: x[1], reverse=True)
            
            logger.info(f"Reranking complete. Top score: {normalized_scores.max():.4f}")
            
            return ranked[:top_k]
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Using original order.")
            return [(idx, 1.0) for idx in candidate_indices[:top_k]]
    
    def retrieve(self, query: str, query_embedding: np.ndarray, 
                 top_k: int = 10, use_reranking: bool = True) -> List[Dict]:
        """
        Main retrieval function with optional reranking.
        """
        # Step 1: Hybrid search
        hybrid_results = self.hybrid_search(query, query_embedding, top_k=30)
        candidate_indices = [idx for idx, _ in hybrid_results]
        
        # Step 2: Reranking (optional but recommended)
        if use_reranking and self.reranker_available:
            reranked_results = self.rerank(query, candidate_indices, top_k=top_k)
            final_indices = [idx for idx, _ in reranked_results]
            final_scores = [score for _, score in reranked_results]
        else:
            final_indices = candidate_indices[:top_k]
            final_scores = [score for _, score in hybrid_results[:top_k]]
        
        # Step 3: Prepare final results
        results = []
        for idx, score in zip(final_indices, final_scores):
            assessment = self.assessments[idx].copy()
            # Ensure score is in [0, 1] range
            assessment['score'] = float(max(0.0, min(1.0, score)))
            results.append(assessment)
        
        return results


