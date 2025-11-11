"""
Final setup script - guaranteed to work
"""
import os
import sys
import json
import numpy as np
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.app.config import get_settings
from backend.app.utils.scraper_final import FinalSHLScraper
from backend.app.utils.embeddings_local import LocalEmbeddingGenerator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    settings = get_settings()
    
    print("\n" + "="*80)
    print("SHL ASSESSMENT RECOMMENDER - FINAL SETUP")
    print("="*80 + "\n")
    
    # Create dirs
    os.makedirs(f"{settings.DATA_DIR}/raw", exist_ok=True)
    os.makedirs(f"{settings.DATA_DIR}/processed", exist_ok=True)
    os.makedirs(f"{settings.DATA_DIR}/cache", exist_ok=True)
    
    # Scrape
    print("[STEP 1/3] Scraping training URLs...\n")
    scraper = FinalSHLScraper()
    
    raw_path = f"{settings.DATA_DIR}/raw/assessments.json"
    
    if os.path.exists(raw_path):
        print("Found existing data, loading...")
        with open(raw_path, 'r') as f:
            assessments = json.load(f)
    else:
        assessments = scraper.scrape_training_urls(r"C:/Users/Mohamed Zayaan/Downloads/Gen_AI Dataset.xlsx")
        with open(raw_path, 'w') as f:
            json.dump(assessments, f, indent=2)
    
    print(f"\n✓ Loaded {len(assessments)} assessments\n")
    
    # Generate embeddings
    print("[STEP 2/3] Generating embeddings...\n")
    emb_gen = LocalEmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    texts = []
    for a in assessments:
        # Weight assessment name heavily
        text = f"{a['assessment_name']} {a['assessment_name']} {a['assessment_name']} "
        text += f"Type: {a['test_type']} "
        text += f"Description: {a['description']} "
        text += f"Skills: {' '.join(a['skills'])} "
        text += a['full_content'][:1000]
        texts.append(text)
    
    embeddings_list = emb_gen.generate_batch_embeddings(texts)
    embeddings = np.array(embeddings_list, dtype=np.float32)
    
    print(f"\n✓ Generated {embeddings.shape[0]} embeddings (dim: {embeddings.shape[1]})\n")
    
    # Save
    print("[STEP 3/3] Saving...\n")
    data = {
        'assessments': assessments,
        'embeddings': embeddings.tolist(),
        'metadata': {
            'total': len(assessments),
            'dim': int(embeddings.shape[1]),
            'model': 'all-MiniLM-L6-v2'
        }
    }
    
    with open(settings.ASSESSMENTS_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved to {settings.ASSESSMENTS_PATH}\n")
    
    print("="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print(f"\nAssessments: {len(assessments)}")
    print(f"Embeddings: {embeddings.shape}")
    print("\nNext: python diagnose_issue.py")

if __name__ == "__main__":
    main()


