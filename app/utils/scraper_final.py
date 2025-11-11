"""
FIXED Final scraper - scrapes EXACT URLs from training data
"""
import requests
from bs4 import BeautifulSoup
import json
import time
import re
import logging
import pandas as pd
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalSHLScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def fetch_url(self, url: str) -> Optional[str]:
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Failed: {url[:50]}... - {e}")
            return None
    
    def extract_name_from_url(self, url: str) -> str:
        parts = url.rstrip('/').split('/')
        name = parts[-1].replace('-new', '').replace('-', ' ')
        name = ' '.join(w.capitalize() for w in name.split() if w and not w.isdigit())
        return name or "Assessment"
    
    def parse_page(self, url: str, html: Optional[str]) -> Dict:
        assessment = {
            'url': url,
            'assessment_name': self.extract_name_from_url(url),
            'description': f'Assessment: {self.extract_name_from_url(url)}',
            'test_type': 'Assessment',
            'duration': '',
            'skills': [],
            'full_content': url
        }
        
        if not html:
            return assessment
        
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            # Title
            for tag in ['h1', 'h2', 'title']:
                elem = soup.find(tag)
                if elem:
                    text = elem.get_text(strip=True)
                    text = re.sub(r'\s*\|\s*SHL.*', '', text)
                    if text and len(text) > 3:
                        assessment['assessment_name'] = text
                        break
            
            # Description
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')[:5]]
            paragraphs = [p for p in paragraphs if len(p) > 50]
            if paragraphs:
                assessment['description'] = ' '.join(paragraphs)[:500]
            
            # Full content
            all_text = soup.get_text(separator=' ', strip=True)
            assessment['full_content'] = ' '.join(all_text.split())[:5000]
            
            # Duration
            if match := re.search(r'(\d+)\s*(minute|min|hour|hr)', all_text, re.I):
                assessment['duration'] = match.group(0)
            
            # Type
            text_lower = all_text.lower()
            if any(k in text_lower for k in ['java', 'python', 'programming', 'code', 'software']):
                assessment['test_type'] = 'Technical'
            elif any(k in text_lower for k in ['cognitive', 'reasoning', 'aptitude']):
                assessment['test_type'] = 'Cognitive'
            elif any(k in text_lower for k in ['personality', 'behavioral', 'behaviour']):
                assessment['test_type'] = 'Behavioral'
            
            # Skills
            keywords = ['java', 'python', 'javascript', 'c++', '.net', 'sql',
                       'leadership', 'communication', 'teamwork', 'analytical']
            for kw in keywords:
                if kw in text_lower:
                    assessment['skills'].append(kw.title())
                    if len(assessment['skills']) >= 5:
                        break
        
        except Exception as e:
            logger.error(f"Parse error: {e}")
        
        return assessment
    
    def scrape_training_urls(self, excel_path: str, output_path: str) -> List[Dict]:
        """
        FIXED: Main scraping function with output_path parameter
        """
        logger.info("="*80)
        logger.info("SCRAPING EXACT TRAINING URLs")
        logger.info("="*80)
        
        # Load training URLs
        df = pd.read_excel(excel_path, sheet_name='Train-Set')
        urls = df['Assessment_url'].unique().tolist()
        
        logger.info(f"\nURLs to scrape: {len(urls)}\n")
        
        assessments = []
        for idx, url in enumerate(urls, 1):
            logger.info(f"[{idx}/{len(urls)}] {url[:70]}...")
            
            html = self.fetch_url(url)
            assessment = self.parse_page(url, html)
            assessments.append(assessment)
            
            logger.info(f"  ✓ {assessment['assessment_name']}")
            time.sleep(0.5)
        
        # Save to output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(assessments, f, indent=2, ensure_ascii=False)
        
        logger.info("\n" + "="*80)
        logger.info(f"DONE! Scraped {len(assessments)}/{len(urls)} assessments")
        logger.info(f"Saved to: {output_path}")
        logger.info("="*80)
        
        return assessments

