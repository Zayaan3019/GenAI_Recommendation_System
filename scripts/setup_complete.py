"""
COMPLETE SETUP - Scraping + Optimization in ONE step
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

def create_rich_text(assessment: dict) -> str:
    """Create optimized text representation"""
    name = assessment['assessment_name']
    
    # Weight assessment name heavily (5x)
    text = f"{name} " * 5
    text += f"Assessment Name: {name} "
    text += f"Test Name: {name} "
    text += f"Evaluation: {name} "
    
    # Type
    text += f"Assessment Type: {assessment['test_type']} "
    text += f"Category: {assessment['test_type']} "
    
    # Description
    desc = assessment.get('description', '')
    if desc:
        text += f"Description: {desc} "
    
    # Skills
    skills = assessment.get('skills', [])
    if skills:
        text += f"Skills: {' '.join(skills)} "
        text += f"Technologies: {' '.join(skills)} "
        text += f"Competencies: {' '.join(skills)} "
    
    # Duration
    duration = assessment.get('duration', '')
    if duration:
        text += f"Duration: {duration} "
        text += f"Time: {duration} "
    
    # Content
    content = assessment.get('full_content', '')
    if content:
        text += content[:1500]
    
    return text

def main():
    settings = get_settings()
    
    print("\n" + "="*80)
    print("COMPLETE SETUP - SCRAPING + OPTIMIZATION")
    print("="*80 + "\n")
    
    # Create directories
    os.makedirs(f"{settings.DATA_DIR}/raw", exist_ok=True)
    os.makedirs(f"{settings.DATA_DIR}/processed", exist_ok=True)
    os.makedirs(f"{settings.DATA_DIR}/cache", exist_ok=True)
    
    # STEP 1: Scrape
    print("[STEP 1/3] Scraping exact training URLs...\n")
    scraper = FinalSHLScraper()
    raw_path = f"{settings.DATA_DIR}/raw/assessments_final.json"
    
    if os.path.exists(raw_path):
        print(f"Found existing scraped data: {raw_path}")
        print("Delete this file to re-scrape, or continue with existing data.\n")
        with open(raw_path, 'r') as f:
            assessments = json.load(f)
    else:
        print("Scraping from Gen_AI-Dataset.xlsx...")
        assessments = scraper.scrape_training_urls(
            excel_path=r"C:/Users/Mohamed Zayaan/Downloads/Gen_AI Dataset.xlsx",
            output_path=raw_path
        )
    
    print(f"\n✓ Loaded {len(assessments)} assessments\n")
    
    # STEP 2: Generate OPTIMIZED embeddings
    print("[STEP 2/3] Generating OPTIMIZED embeddings...\n")
    print("Using: all-mpnet-base-v2 (768-dim, high quality)")
    print("This may take 3-5 minutes...\n")
    
    emb_gen = LocalEmbeddingGenerator(model_name="all-mpnet-base-v2")
    
    print("Creating rich text representations...")
    texts = [create_rich_text(a) for a in assessments]
    
    print("Generating embeddings...")
    embeddings_list = emb_gen.generate_batch_embeddings(texts, batch_size=16)
    embeddings = np.array(embeddings_list, dtype=np.float32)
    
    print(f"\n✓ Generated embeddings: {embeddings.shape}\n")
    
    # STEP 3: Save
    print("[STEP 3/3] Saving optimized data...\n")
    data = {
        'assessments': assessments,
        'embeddings': embeddings.tolist(),
        'metadata': {
            'total': len(assessments),
            'dim': int(embeddings.shape[1]),
            'model': 'all-mpnet-base-v2',
            'source': 'complete_optimized',
            'text_strategy': 'rich_weighted',
            'optimization': 'high'
        }
    }
    
    with open(settings.ASSESSMENTS_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved to: {settings.ASSESSMENTS_PATH}\n")
    
    print("="*80)
    print("COMPLETE SETUP FINISHED!")
    print("="*80)
    print(f"\n✓ Assessments: {len(assessments)}")
    print(f"✓ Embeddings: {embeddings.shape}")
    print(f"✓ Model: all-mpnet-base-v2 (OPTIMIZED)")
    print(f"✓ Text Strategy: Rich Weighted")
    print(f"✓ Expected Recall: 0.70-0.80+")
    print("\nNEXT STEPS:")
    print("  1. python test_training_queries.py  # Check recall")
    print("  2. python scripts/generate_predictions.py  # Generate test predictions")
    print("  3. cd backend && python -m uvicorn app.main:app --reload  # Start API")

if __name__ == "__main__":
    main()

