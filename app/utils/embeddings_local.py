from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from functools import lru_cache
import hashlib
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class LocalEmbeddingGenerator:
    """
    Generate embeddings using local Sentence Transformers model with model-specific caching
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "./data/cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key that includes model name to avoid conflicts"""
        # Include model name in hash to separate caches for different models
        combined = f"{self.model_name}:::{text}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str):
        """Load embedding from cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for a single text."""
        cache_key = self._get_cache_key(text)
        
        # Try cache first
        if use_cache:
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                # Verify dimension matches
                if cached_embedding.shape[0] == self.embedding_dim:
                    return cached_embedding
                else:
                    logger.warning(f"Cached embedding dimension mismatch: {cached_embedding.shape[0]} != {self.embedding_dim}, regenerating...")
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        embedding = embedding.astype(np.float32)
        
        # Verify dimension
        if embedding.shape[0] != self.embedding_dim:
            logger.error(f"Generated embedding has wrong dimension: {embedding.shape[0]} != {self.embedding_dim}")
        
        # Cache result
        if use_cache:
            self._save_to_cache(cache_key, embedding)
        
        return embedding
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding optimized for queries."""
        return self.generate_embedding(query)
    
    def generate_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batches."""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=batch_size
            )
            
            for emb in batch_embeddings:
                embeddings.append(emb.astype(np.float32))
        
        return embeddings

