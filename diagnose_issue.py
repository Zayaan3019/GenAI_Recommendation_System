"""
Diagnose the recall issue
"""
import pandas as pd
import json
from backend.app.config import get_settings

# Load training data
train_df = pd.read_excel(r"C:/Users/Mohamed Zayaan/Downloads/Gen_AI Dataset.xlsx", sheet_name='Train-Set')

print("="*80)
print("DIAGNOSTIC REPORT")
print("="*80)

# Check training data URLs
print("\n1. TRAINING DATA URLS (Sample):")
print("-" * 80)
for url in train_df['Assessment_url'].unique()[:5]:
    print(f"  {url}")

# Load scraped assessments
settings = get_settings()
try:
    with open(settings.ASSESSMENTS_PATH, 'r') as f:
        data = json.load(f)
    
    scraped_urls = [a['url'] for a in data['assessments']]
    
    print(f"\n2. SCRAPED ASSESSMENTS: {len(scraped_urls)}")
    print("-" * 80)
    for url in scraped_urls[:5]:
        print(f"  {url}")
    
    # Check overlap
    train_urls = set(train_df['Assessment_url'].str.lower().str.rstrip('/'))
    scraped_urls_set = set([u.lower().rstrip('/') for u in scraped_urls])
    
    overlap = train_urls.intersection(scraped_urls_set)
    
    print(f"\n3. URL MATCHING:")
    print("-" * 80)
    print(f"  Training unique URLs: {len(train_urls)}")
    print(f"  Scraped URLs: {len(scraped_urls_set)}")
    print(f"  Overlap: {len(overlap)}")
    print(f"  Coverage: {len(overlap)/len(train_urls)*100:.1f}%")
    
    if len(overlap) == 0:
        print("\n❌ PROBLEM FOUND: NO URL OVERLAP!")
        print("   Training URLs and scraped URLs don't match at all.")
        print("\n   Training URL pattern:")
        print(f"   {list(train_urls)[0]}")
        print("\n   Scraped URL pattern:")
        print(f"   {list(scraped_urls_set)[0]}")
    
except FileNotFoundError:
    print("❌ Assessments file not found!")

print("\n" + "="*80)
