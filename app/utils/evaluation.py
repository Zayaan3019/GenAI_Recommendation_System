import pandas as pd
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """
    Evaluate recommendation system using Mean Recall@K metric as specified in assignment.
    """
    
    def __init__(self, ground_truth_path: str):
        """
        Load ground truth data for evaluation.
        
        Args:
            ground_truth_path: Path to Excel file with train/test sets
        """
        self.train_df = pd.read_excel(ground_truth_path, sheet_name='Train-Set')
        self.test_df = pd.read_excel(ground_truth_path, sheet_name='Test-Set')
        
        # Create ground truth dictionary
        self.ground_truth = {}
        for query in self.train_df['Query'].unique():
            relevant_urls = self.train_df[self.train_df['Query'] == query]['Assessment_url'].tolist()
            # Normalize URLs (remove trailing slashes, lowercase)
            relevant_urls = [url.rstrip('/').lower() for url in relevant_urls]
            self.ground_truth[query] = set(relevant_urls)
        
        logger.info(f"Loaded ground truth: {len(self.ground_truth)} queries, "
                   f"{len(self.train_df)} relevant assessments")
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        return url.rstrip('/').lower()
    
    def recall_at_k(self, query: str, recommended_urls: List[str], k: int = 10) -> float:
        """
        Calculate Recall@K for a single query.
        
        Recall@K = (# of relevant items in top-K) / (total # of relevant items)
        """
        if query not in self.ground_truth:
            logger.warning(f"Query not in ground truth: {query[:50]}...")
            return 0.0
        
        relevant_items = self.ground_truth[query]
        recommended_set = set([self.normalize_url(url) for url in recommended_urls[:k]])
        
        hits = len(relevant_items.intersection(recommended_set))
        total_relevant = len(relevant_items)
        
        recall = hits / total_relevant if total_relevant > 0 else 0.0
        
        return recall
    
    def mean_recall_at_k(self, predictions: Dict[str, List[str]], k: int = 10) -> Tuple[float, Dict]:
        """
        Calculate Mean Recall@K across all queries.
        
        Args:
            predictions: Dict mapping queries to lists of recommended URLs
            k: Number of recommendations to consider
            
        Returns:
            mean_recall: Average recall across all queries
            details: Per-query recall scores
        """
        recalls = []
        details = {}
        
        for query, recommended_urls in predictions.items():
            recall = self.recall_at_k(query, recommended_urls, k)
            recalls.append(recall)
            details[query] = recall
        
        mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
        
        logger.info(f"Mean Recall@{k}: {mean_recall:.4f}")
        return mean_recall, details
    
    def evaluate_on_train(self, retriever_func, k: int = 10) -> Tuple[float, Dict]:
        """
        Evaluate retriever on training set.
        
        Args:
            retriever_func: Function that takes a query and returns list of URLs
            k: Number of recommendations
        """
        predictions = {}
        
        for query in self.ground_truth.keys():
            recommended_urls = retriever_func(query, top_k=k)
            predictions[query] = recommended_urls
        
        return self.mean_recall_at_k(predictions, k)
    
    def generate_test_predictions(self, retriever_func, k: int = 10, 
                                  output_path: str = "predictions.csv") -> pd.DataFrame:
        """
        Generate predictions for test set and save to CSV.
        
        Args:
            retriever_func: Function that takes a query and returns list of assessment dicts
            k: Number of recommendations
            output_path: Path to save predictions CSV
        """
        predictions = []
        
        for idx, row in self.test_df.iterrows():
            query = row['Query']
            logger.info(f"Generating prediction {idx+1}/{len(self.test_df)}: {query[:50]}...")
            
            # Get recommendations
            recommended_assessments = retriever_func(query, top_k=k)
            
            # Extract URLs
            for assessment in recommended_assessments:
                predictions.append({
                    'Query': query,
                    'Assessment_url': assessment['url']
                })
        
        # Create DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Save to CSV
        pred_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        return pred_df

