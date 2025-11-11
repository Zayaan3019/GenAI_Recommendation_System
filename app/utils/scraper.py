import requests
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict, Optional
import re
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHLCatalogScraper:
    """
    Advanced web scraper for SHL product catalog with robust error handling
    and data extraction capabilities.
    """
    
    def __init__(self, base_url: str = "https://www.shl.com/solutions/products/product-catalog/"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch webpage content with retry logic."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_assessment_links(self, html: str) -> List[str]:
        """Extract individual assessment URLs from catalog page."""
        soup = BeautifulSoup(html, 'lxml')
        links = []
        
        # Find all assessment links (adjust selectors based on actual HTML structure)
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/product-catalog/view/' in href and href not in links:
                # Ensure full URL
                if not href.startswith('http'):
                    href = f"https://www.shl.com{href}"
                links.append(href)
        
        logger.info(f"Found {len(links)} assessment links")
        return links
    
    def parse_assessment_page(self, url: str, html: str) -> Optional[Dict]:
        """Parse individual assessment page to extract detailed information."""
        soup = BeautifulSoup(html, 'lxml')
        
        try:
            assessment = {
                'url': url,
                'assessment_name': '',
                'description': '',
                'test_type': '',
                'duration': '',
                'skills': [],
                'target_audience': '',
                'key_features': []
            }
            
            # Extract title
            title_elem = soup.find('h1') or soup.find('h2', class_=re.compile('title|heading|product'))
            if title_elem:
                assessment['assessment_name'] = title_elem.get_text(strip=True)
            
            # Extract description (multiple possible locations)
            desc_elem = (soup.find('div', class_=re.compile('description|overview|summary')) or
                        soup.find('p', class_=re.compile('intro|lead')))
            if desc_elem:
                assessment['description'] = desc_elem.get_text(strip=True)
            
            # Extract metadata sections
            for section in soup.find_all(['div', 'section']):
                section_text = section.get_text(strip=True).lower()
                
                # Duration
                if 'duration' in section_text or 'time' in section_text:
                    duration_match = re.search(r'(\d+)\s*(minute|min|hour|hr)', section_text, re.IGNORECASE)
                    if duration_match:
                        assessment['duration'] = duration_match.group(0)
                
                # Test type
                if 'type' in section_text or 'category' in section_text:
                    for test_type in ['cognitive', 'personality', 'behavioral', 'technical', 'skills']:
                        if test_type in section_text:
                            assessment['test_type'] = test_type.capitalize()
                            break
            
            # Extract skills/competencies
            skills_section = soup.find(['ul', 'div'], class_=re.compile('skill|competenc|measure'))
            if skills_section:
                for li in skills_section.find_all('li'):
                    skill = li.get_text(strip=True)
                    if skill:
                        assessment['skills'].append(skill)
            
            # Extract all text content for semantic search
            all_text = soup.get_text(separator=' ', strip=True)
            assessment['full_content'] = ' '.join(all_text.split())[:5000]  # Limit length
            
            # Extract assessment name from URL if not found
            if not assessment['assessment_name']:
                assessment['assessment_name'] = url.split('/')[-2].replace('-', ' ').title()
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return None
    
    def scrape_catalog(self, save_path: str = "./data/raw/assessments.json") -> List[Dict]:
        """
        Main scraping function to extract all assessments from catalog.
        """
        logger.info("Starting SHL catalog scraping...")
        
        # Step 1: Get catalog main page
        catalog_html = self.fetch_page(self.base_url)
        if not catalog_html:
            logger.error("Failed to fetch catalog page")
            return []
        
        # Step 2: Extract all assessment links
        assessment_urls = self.extract_assessment_links(catalog_html)
        
        # Step 3: Scrape each assessment page
        assessments = []
        for idx, url in enumerate(assessment_urls, 1):
            logger.info(f"Scraping assessment {idx}/{len(assessment_urls)}: {url}")
            
            html = self.fetch_page(url)
            if html:
                assessment_data = self.parse_assessment_page(url, html)
                if assessment_data:
                    assessments.append(assessment_data)
            
            # Rate limiting
            time.sleep(0.5)
        
        # Step 4: Save results
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(assessments, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scraping complete! Saved {len(assessments)} assessments to {save_path}")
        return assessments
    
    def load_assessments(self, path: str = "./data/raw/assessments.json") -> List[Dict]:
        """Load previously scraped assessments."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return []
