from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from enum import Enum

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Job description or natural language query")
    top_k: Optional[int] = Field(10, ge=1, le=10, description="Number of recommendations")

class Assessment(BaseModel):
    assessment_name: str
    url: str
    score: float
    test_type: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None
    skills: Optional[List[str]] = None

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[Assessment]
    total_results: int
    processing_time_ms: float
    metadata: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    timestamp: str
