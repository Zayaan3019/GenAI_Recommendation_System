from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path
from typing import Optional

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent
BACKEND_DIR = Path(__file__).parent.parent

class Settings(BaseSettings):
    # API Keys
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API Key")
    
    # Application
    APP_NAME: str = "SHL Assessment Recommender"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Retrieval Parameters
    TOP_K_RETRIEVAL: int = 30
    TOP_K_RERANK: int = 10
    MIN_RECOMMENDATIONS: int = 5
    MAX_RECOMMENDATIONS: int = 10
    
    # Embedding
    EMBEDDING_MODEL: str = "models/embedding-001"
    EMBEDDING_DIM: int = 768
    
    # Paths (relative to backend directory)
    DATA_DIR: str = str(BACKEND_DIR / "data")
    ASSESSMENTS_PATH: str = str(BACKEND_DIR / "data" / "processed" / "assessments.json")
    VECTORDB_PATH: str = str(BACKEND_DIR / "data" / "vectors")
    
    # Cache
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600
    
    # URLs
    SHL_CATALOG_URL: str = "https://www.shl.com/solutions/products/product-catalog/"
    
    class Config:
        # Look for .env file in backend directory
        env_file = str(BACKEND_DIR / ".env")
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields

@lru_cache()
def get_settings():
    return Settings()

