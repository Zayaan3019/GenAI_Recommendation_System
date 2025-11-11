from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class RecommendationBalancer:
    """
    Ensures balanced recommendations across different assessment types
    as per assignment requirements.
    """
    
    def __init__(self, min_recommendations: int = 5, max_recommendations: int = 10):
        self.min_recommendations = min_recommendations
        self.max_recommendations = max_recommendations
    
    def balance_recommendations(self, recommendations: List[Dict], 
                               target_count: int = 10) -> List[Dict]:
        """
        Balance recommendations to include diverse assessment types.
        
        Strategy:
        1. Identify assessment categories (technical, behavioral, cognitive, etc.)
        2. Ensure representation from multiple categories
        3. Maintain score-based ranking while diversifying
        """
        if len(recommendations) <= target_count:
            return recommendations
        
        # Categorize assessments
        categorized = self._categorize_assessments(recommendations)
        
        # Build balanced list
        balanced = []
        used_indices = set()
        
        # First pass: Take top from each category
        max_per_category = max(2, target_count // len(categorized))
        for category, assessments in categorized.items():
            for assessment in assessments[:max_per_category]:
                if len(balanced) < target_count:
                    idx = recommendations.index(assessment)
                    if idx not in used_indices:
                        balanced.append(assessment)
                        used_indices.add(idx)
        
        # Second pass: Fill remaining slots with highest scores
        for idx, assessment in enumerate(recommendations):
            if len(balanced) >= target_count:
                break
            if idx not in used_indices:
                balanced.append(assessment)
                used_indices.add(idx)
        
        logger.info(f"Balanced recommendations: {len(balanced)} assessments from {len(categorized)} categories")
        return balanced[:target_count]
    
    def _categorize_assessments(self, recommendations: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize assessments by type."""
        categories = {
            'technical': [],
            'behavioral': [],
            'cognitive': [],
            'personality': [],
            'other': []
        }
        
        for assessment in recommendations:
            test_type = assessment.get('test_type', '').lower()
            name = assessment.get('assessment_name', '').lower()
            
            # Categorize based on keywords
            if any(kw in test_type + name for kw in ['java', 'python', 'code', 'programming', 'technical', 'software']):
                categories['technical'].append(assessment)
            elif any(kw in test_type + name for kw in ['behavior', 'situational', 'judgment']):
                categories['behavioral'].append(assessment)
            elif any(kw in test_type + name for kw in ['cognitive', 'reasoning', 'aptitude', 'ability']):
                categories['cognitive'].append(assessment)
            elif any(kw in test_type + name for kw in ['personality', 'trait', 'style']):
                categories['personality'].append(assessment)
            else:
                categories['other'].append(assessment)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
