import google.generativeai as genai
from typing import List, Optional
import numpy as np
from functools import lru_cache
import hashlib
import pickle
import os
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generate embeddings using Google's Gemini API with caching and batch processing.
    """
    
    def __init__(self, api_key: str, model: str = "models/embedding-001", cache_dir: str = "./data/cache"):
        genai.configure(api_key=api_key)
        self.model = model
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for a single text."""
        cache_key = self._get_cache_key(text)
        
        # Try cache first
        if use_cache:
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate embedding
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embedding = np.array(result['embedding'], dtype=np.float32)
            
            # Cache result
            if use_cache:
                self._save_to_cache(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding optimized for queries."""
        try:
            result = genai.embed_content(
                model=self.model,
                content=query,
                task_type="retrieval_query"
            )
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def generate_batch_embeddings(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            for text in batch:
                emb = self.generate_embedding(text)
                embeddings.append(emb)
        
        return embeddings
