#!/usr/bin/env python3
"""
Wikipedia Domain Dataset Crawler
Crawls Wikipedia Sports and Politics categories, extracts keywords, and builds labeled sentence datasets.
"""

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import csv
import time
import re
from collections import defaultdict
from typing import List, Set, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WikipediaCrawler:
    """Crawls Wikipedia categories and extracts domain-specific sentences."""
    
    def __init__(self):
        self.base_url = "https://en.wikipedia.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Research Bot)'
        })
        
        # Download required NLTK data
        self._setup_nltk()
        
        self.stopwords = set(stopwords.words('english'))
        
    def _setup_nltk(self):
        """Download required NLTK data packages."""
        required_packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)
    
    def get_category_pages(self, category: str, max_pages: int = 100) -> List[str]:
        """
        Get article URLs from a Wikipedia category.
        
        Args:
            category: Category name (e.g., 'Sports' or 'Politics')
            max_pages: Maximum number of pages to retrieve
            
        Returns:
            List of article URLs
        """
        logger.info(f"Fetching pages from category: {category}")
        pages = []
        
        # Wikipedia category URL
        category_url = f"{self.base_url}/wiki/Category:{category}"
        
        try:
            response = self.session.get(category_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all article links in the category
            content_div = soup.find('div', {'id': 'mw-pages'})
            if content_div:
                links = content_div.find_all('a')
                for link in links[:max_pages]:
                    href = link.get('href')
                    if href and href.startswith('/wiki/') and ':' not in href:
                        pages.append(self.base_url + href)
            
            # Also get subcategories and their articles
            subcategories = soup.find('div', {'id': 'mw-subcategories'})
            if subcategories and len(pages) < max_pages:
                subcat_links = subcategories.find_all('a')[:5]  # Limit subcategories
                for subcat in subcat_links:
                    href = subcat.get('href')
                    if href and '/wiki/Category:' in href:
                        subcat_name = href.split('Category:')[-1]
                        subpages = self.get_category_pages(subcat_name, max_pages=20)
                        pages.extend(subpages)
                        if len(pages) >= max_pages:
                            break
                            
        except Exception as e:
            logger.error(f"Error fetching category {category}: {e}")
        
        return list(set(pages))[:max_pages]  # Remove duplicates
    
    def extract_text_from_page(self, url: str) -> str:
        """
        Extract main text content from a Wikipedia page.
        
        Args:
            url: Wikipedia article URL
            
        Returns:
            Extracted text content
        """
        try:
            time.sleep(0.5)  # Be respectful to Wikipedia servers
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get main content
            content = soup.find('div', {'id': 'mw-content-text'})
            if not content:
                return ""
            
            # Remove unwanted elements
            for tag in content.find_all(['script', 'style', 'table', 'div.reflist', 
                                         'div.navbox', 'div.infobox']):
                tag.decompose()
            
            # Extract paragraphs
            paragraphs = content.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            
            # Clean text
            text = re.sub(r'\[.*?\]', '', text)  # Remove citation numbers
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from {url}: {e}")
            return ""
    
    def extract_keywords(self, texts: List[str], max_keywords: int = 500) -> List[str]:
        """
        Extract domain keywords using NLTK.
        
        Args:
            texts: List of text documents
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        logger.info("Extracting keywords...")
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Tokenize
        tokens = word_tokenize(combined_text.lower())
        
        # Filter tokens
        filtered_tokens = [
            word for word in tokens
            if word.isalpha()  # Only alphabetic
            and len(word) > 3  # Longer than 3 characters
            and word not in self.stopwords  # Not a stopword
        ]
        
        # Calculate frequency distribution
        freq_dist = FreqDist(filtered_tokens)
        
        # Get most common keywords
        keywords = [word for word, freq in freq_dist.most_common(max_keywords)]
        
        logger.info(f"Extracted {len(keywords)} keywords")
        return keywords
    
    def extract_sentences(self, texts: List[str], min_length: int = 40) -> List[str]:
        """
        Extract sentences from texts, filtering out short ones.
        
        Args:
            texts: List of text documents
            min_length: Minimum character length for sentences
            
        Returns:
            List of valid sentences
        """
        all_sentences = []
        
        for text in texts:
            sentences = sent_tokenize(text)
            
            for sentence in sentences:
                # Clean sentence
                sentence = sentence.strip()
                
                # Filter criteria
                if (len(sentence) >= min_length and
                    not sentence.startswith('http') and
                    sentence.count(' ') >= 5):  # At least 6 words
                    all_sentences.append(sentence)
        
        return all_sentences
    
    def crawl_category(self, category: str, max_pages: int = 100, 
                      max_sentences: int = 50000, max_keywords: int = 500) -> Tuple[List[str], List[str]]:
        """
        Crawl a Wikipedia category and extract keywords and sentences.
        
        Args:
            category: Category name
            max_pages: Maximum pages to crawl
            max_sentences: Target number of sentences
            max_keywords: Maximum keywords to extract
            
        Returns:
            Tuple of (keywords, sentences)
        """
        logger.info(f"Starting crawl for category: {category}")
        
        # Get article URLs
        urls = self.get_category_pages(category, max_pages)
        logger.info(f"Found {len(urls)} article URLs")
        
        # Extract texts
        texts = []
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing page {i}/{len(urls)}: {url}")
            text = self.extract_text_from_page(url)
            if text:
                texts.append(text)
        
        logger.info(f"Extracted text from {len(texts)} pages")
        
        # Extract keywords
        keywords = self.extract_keywords(texts, max_keywords)
        
        # Extract sentences
        sentences = self.extract_sentences(texts)
        
        # Limit to max_sentences
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        logger.info(f"Collected {len(sentences)} sentences for {category}")
        
        return keywords, sentences


def create_dataset(categories: Dict[str, str], output_file: str = 'wikipedia_dataset.csv'):
    """
    Create a labeled dataset from multiple Wikipedia categories.
    
    Args:
        categories: Dictionary mapping category names to labels
        output_file: Output CSV file path
    """
    crawler = WikipediaCrawler()
    
    # Collect data for each category
    dataset = []
    all_keywords = {}
    
    for category, label in categories.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing category: {category} (label: {label})")
        logger.info(f"{'='*60}\n")
        
        keywords, sentences = crawler.crawl_category(
            category=category,
            max_pages=100,
            max_sentences=50000,
            max_keywords=500
        )
        
        all_keywords[label] = keywords
        
        # Add sentences to dataset with label
        for sentence in sentences:
            dataset.append({
                'text': sentence,
                'label': label,
                'category': category
            })
    
    # Save dataset to CSV
    logger.info(f"\n{'='*60}")
    logger.info(f"Saving dataset to {output_file}")
    logger.info(f"{'='*60}\n")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if dataset:
            writer = csv.DictWriter(f, fieldnames=['text', 'label', 'category'])
            writer.writeheader()
            writer.writerows(dataset)
    
    logger.info(f"Dataset saved with {len(dataset)} sentences")
    
    # Save keywords separately
    keywords_file = output_file.replace('.csv', '_keywords.csv')
    with open(keywords_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'keyword', 'rank'])
        
        for label, keywords in all_keywords.items():
            for rank, keyword in enumerate(keywords, 1):
                writer.writerow([label, keyword, rank])
    
    logger.info(f"Keywords saved to {keywords_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    for label in all_keywords.keys():
        count = sum(1 for row in dataset if row['label'] == label)
        print(f"{label}: {count} sentences, {len(all_keywords[label])} keywords")
    print(f"\nTotal sentences: {len(dataset)}")
    print("="*60)


def main():
    """Main execution function."""
    
    # Define categories to crawl
    categories = {
        'Sports': 'sports',
        'Politics': 'politics'
    }
    
    # Create dataset
    create_dataset(categories, output_file='wikipedia_dataset.csv')
    
    logger.info("\n✓ Dataset creation complete!")


if __name__ == '__main__':
    main()
