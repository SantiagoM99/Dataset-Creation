#!/usr/bin/env python3
"""
Spanish Medical Article Collector v1.0
Collects copyright-free Spanish medical articles with abstracts and MeSH terms.
Breaks the 9999 article limit using multiple search strategies.

Requirements:
- Articles must have abstracts
- Articles must have MeSH terms  
- Articles must be copyright-free (open access/free full text)
- Language: Spanish
"""

import requests
import xml.etree.ElementTree as ET
import json
import os
import time
import argparse
from datetime import datetime
import re
from collections import defaultdict
from typing import List, Dict, Optional
import logging

class SpanishArticleCollector:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Spanish Article Collector
        
        Args:
            api_key: NCBI API key for higher rate limits (optional)
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Spanish-Article-Collector/1.0 (Python)'
        })
        
        self.api_key = api_key
        self.pubmed_search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.pubmed_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        
        # Rate limiting: 3 requests/second with API key, 2/second without
        self.rate_limit_delay = 0.34 if api_key else 0.5
        
        # Collection statistics
        self.stats = {
            'total_searched': 0,
            'total_found': 0,
            'filtered_no_abstract': 0,
            'filtered_no_mesh': 0,
            'final_articles': 0
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('spanish_collector.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _add_api_key(self, params: Dict) -> Dict:
        """Add API key to request parameters if available"""
        if self.api_key:
            params['api_key'] = self.api_key
        return params
    
    def _rate_limit(self):
        """Apply rate limiting to respect NCBI guidelines"""
        time.sleep(self.rate_limit_delay)
    
    def _build_search_query(self, base_query_parts: List[str]) -> str:
        """
        Build search query with mandatory filters for copyright-free articles
        
        Args:
            base_query_parts: List of base search terms
            
        Returns:
            Complete search query string
        """
        query_parts = ['spanish[Language]']  # Always search for Spanish articles
        query_parts.extend(base_query_parts)
        
        # Mandatory filters
        query_parts.append('hasabstract')  # Must have abstract
        query_parts.append('(free full text[sb] OR "open access"[Filter] OR "pmc open access"[Filter])')  # Copyright-free
        
        return ' AND '.join(query_parts)
    
    def search_by_date_range(self, start_year: int, end_year: int) -> set:
        """
        Search articles by date range
        
        Args:
            start_year: Starting year for search
            end_year: Ending year for search
            
        Returns:
            Set of PMIDs found
        """
        self.logger.info(f"Searching articles from {start_year} to {end_year}")
        all_pmids = set()
        
        for year in range(start_year, end_year + 1):
            year_pmids = self._search_single_year(year)
            all_pmids.update(year_pmids)
            
            if len(year_pmids) > 0:
                self.logger.info(f"Year {year}: {len(year_pmids)} articles found")
            
            self._rate_limit()
        
        return all_pmids
    
    def _search_single_year(self, year: int) -> set:
        """Search articles for a specific year"""
        date_filter = f'"{year}/01/01"[PDAT] : "{year}/12/31"[PDAT]'
        search_query = self._build_search_query([date_filter])
        
        return self._execute_search(search_query)
    
    def search_by_journals(self, journal_list: List[str]) -> set:
        """
        Search articles by specific journals
        
        Args:
            journal_list: List of journal names to search
            
        Returns:
            Set of PMIDs found
        """
        self.logger.info(f"Searching in {len(journal_list)} journals")
        all_pmids = set()
        
        for journal in journal_list:
            journal_filter = f'"{journal}"[Journal]'
            search_query = self._build_search_query([journal_filter])
            
            journal_pmids = self._execute_search(search_query)
            all_pmids.update(journal_pmids)
            
            if len(journal_pmids) > 0:
                self.logger.info(f"Journal '{journal}': {len(journal_pmids)} articles found")
            
            self._rate_limit()
        
        return all_pmids
    
    def search_by_medical_subjects(self, subject_list: List[str]) -> set:
        """
        Search articles by medical subjects/MeSH terms
        
        Args:
            subject_list: List of medical subjects to search
            
        Returns:
            Set of PMIDs found
        """
        self.logger.info(f"Searching by {len(subject_list)} medical subjects")
        all_pmids = set()
        
        for subject in subject_list:
            subject_filter = f'"{subject}"[MeSH Terms]'
            search_query = self._build_search_query([subject_filter])
            
            subject_pmids = self._execute_search(search_query)
            all_pmids.update(subject_pmids)
            
            if len(subject_pmids) > 0:
                self.logger.info(f"Subject '{subject}': {len(subject_pmids)} articles found")
            
            self._rate_limit()
        
        return all_pmids
    
    def _execute_search(self, search_query: str) -> set:
        """
        Execute a PubMed search query
        
        Args:
            search_query: Complete search query string
            
        Returns:
            Set of PMIDs found
        """
        params = self._add_api_key({
            'db': 'pubmed',
            'term': search_query,
            'retmax': 9999,
            'retmode': 'xml'
        })
        
        try:
            response = self.session.get(self.pubmed_search_url, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            pmids = [id_elem.text for id_elem in root.findall('.//Id')]
            
            self.stats['total_searched'] += len(pmids)
            return set(pmids)
            
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return set()
    
    def collect_comprehensive_dataset(self, target_articles: int = 50000) -> set:
        """
        Collect a comprehensive dataset using multiple search strategies
        
        Args:
            target_articles: Target number of articles to collect
            
        Returns:
            Set of unique PMIDs
        """
        self.logger.info("🚀 Starting comprehensive article collection")
        self.logger.info(f"Target: {target_articles:,} copyright-free Spanish articles with abstracts and MeSH terms")
        
        all_pmids = set()
        
        # Strategy 1: Recent years (2000-2025) - best for open access coverage
        if len(all_pmids) < target_articles:
            self.logger.info("📅 Strategy 1: Searching by recent years (2000-2025)")
            date_pmids = self.search_by_date_range(2000, 2025)
            all_pmids.update(date_pmids)
            self.logger.info(f"Date search results: {len(date_pmids):,} new articles (Total: {len(all_pmids):,})")
        
        # Strategy 2: Open access journals
        if len(all_pmids) < target_articles:
            self.logger.info("📚 Strategy 2: Searching open access journals")
            open_access_journals = self._get_open_access_journals()
            journal_pmids = self.search_by_journals(open_access_journals)
            new_pmids = journal_pmids - all_pmids
            all_pmids.update(new_pmids)
            self.logger.info(f"Journal search results: {len(new_pmids):,} new articles (Total: {len(all_pmids):,})")
        
        # Strategy 3: Medical subjects
        if len(all_pmids) < target_articles:
            self.logger.info("🏥 Strategy 3: Searching by medical subjects")
            medical_subjects = self._get_medical_subjects()
            subject_pmids = self.search_by_medical_subjects(medical_subjects)
            new_pmids = subject_pmids - all_pmids
            all_pmids.update(new_pmids)
            self.logger.info(f"Subject search results: {len(new_pmids):,} new articles (Total: {len(all_pmids):,})")
        
        self.stats['total_found'] = len(all_pmids)
        self.logger.info(f"✅ Collection complete: {len(all_pmids):,} unique articles found")
        
        return all_pmids
    
    def _get_open_access_journals(self) -> List[str]:
        """Get list of known open access Spanish/Latin American medical journals"""
        return [
            "Revista de Saude Publica",
            "Cadernos de Saude Publica",
            "Revista Peruana de Medicina Experimental y Salud Publica",
            "Gaceta Sanitaria",
            "Revista Panamericana de Salud Publica",
            "Colombia Medica",
            "Revista Medica de Chile",
            "Acta Medica Colombiana",
            "Biomedica",
            "Medicina Intensiva",
            "Archivos de Bronconeumologia",
            "Anales de Pediatria",
            "Ciencia & Saude Coletiva",
            "Revista Brasileira de Epidemiologia",
            "Interface - Comunicacao Saude Educacao",
            "Revista Española de Cardiologia",
            "Medicina Clinica",
            "Gaceta Medica de Mexico",
            "Salud Publica de Mexico",
            "Revista Medica del Instituto Mexicano del Seguro Social"
        ]
    
    def _get_medical_subjects(self) -> List[str]:
        """Get list of medical subjects for comprehensive coverage"""
        return [
            "Public Health", "Epidemiology", "Cardiology", "Neurology",
            "Oncology", "Pediatrics", "Psychiatry", "Dermatology",
            "Gastroenterology", "Orthopedics", "Ophthalmology", "Gynecology",
            "Urology", "Endocrinology", "Pulmonology", "Nephrology",
            "Rheumatology", "Hematology", "Infectious Diseases",
            "Emergency Medicine", "Family Medicine", "Internal Medicine",
            "Surgery", "Anesthesiology", "Radiology", "Pathology",
            "Pharmacology", "Genetics", "Immunology", "Microbiology",
            "Preventive Medicine", "Occupational Medicine", "Geriatrics",
            "Critical Care Medicine", "Rehabilitation Medicine"
        ]
    
    def fetch_article_details(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch detailed information for articles
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article dictionaries
        """
        self.logger.info(f"📖 Fetching details for {len(pmids):,} articles")
        
        articles = []
        batch_size = 200  # Larger batch size for efficiency
        total_batches = (len(pmids) + batch_size - 1) // batch_size
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch)} articles")
            
            batch_articles = self._fetch_batch_details(batch)
            articles.extend(batch_articles)
            
            self._rate_limit()
            
            # Progress update every 10 batches
            if batch_num % 10 == 0:
                self.logger.info(f"Progress: {len(articles):,} articles processed")
        
        # Filter articles that don't meet our requirements
        filtered_articles = self._apply_final_filters(articles)
        
        self.stats['final_articles'] = len(filtered_articles)
        self.logger.info(f"✅ Final dataset: {len(filtered_articles):,} articles (after filtering)")
        
        return filtered_articles
    
    def _fetch_batch_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch details for a batch of PMIDs"""
        params = self._add_api_key({
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        })
        
        try:
            response = self.session.get(self.pubmed_fetch_url, params=params)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            batch_articles = []
            
            for pubmed_article in root.findall('.//PubmedArticle'):
                article = self._extract_article_data(pubmed_article)
                if article:
                    batch_articles.append(article)
            
            return batch_articles
            
        except Exception as e:
            self.logger.error(f"Error fetching batch: {e}")
            return []
    
    def _extract_article_data(self, article_xml) -> Optional[Dict]:
        """Extract comprehensive article data from XML"""
        try:
            # Basic information
            pmid = self._get_text(article_xml, './/PMID')
            title = self._get_text(article_xml, './/ArticleTitle')
            
            # Abstract (required)
            abstract_info = self._extract_abstract(article_xml)
            if not abstract_info['has_content']:
                self.stats['filtered_no_abstract'] += 1
                return None
            
            # MeSH terms (required)
            mesh_terms = self._extract_mesh_terms(article_xml)
            if not mesh_terms:
                self.stats['filtered_no_mesh'] += 1
                return None
            
            # Additional information
            journal_info = self._extract_journal_info(article_xml)
            authors = self._extract_authors(article_xml)
            publication_info = self._extract_publication_info(article_xml)
            identifiers = self._extract_identifiers(article_xml)
            keywords = self._extract_keywords(article_xml)
            
            # Language detection
            abstract_language = self._detect_language(abstract_info['full_text'])
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract_info,
                'abstract_language': abstract_language,
                'journal': journal_info,
                'authors': authors,
                'mesh_terms': mesh_terms,
                'keywords': keywords,
                'publication_info': publication_info,
                'identifiers': identifiers,
                'collection_date': datetime.now().isoformat(),
                'copyright_status': 'free',  # All articles are pre-filtered for copyright-free status
                'quality_filters_passed': ['has_abstract', 'has_mesh_terms', 'copyright_free']
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting article data: {e}")
            return None
    
    def _get_text(self, element, xpath: str) -> str:
        """Safely extract text from XML element"""
        elem = element.find(xpath)
        if elem is not None:
            return ''.join(elem.itertext()).strip()
        return ''
    
    def _extract_abstract(self, article_xml) -> Dict:
        """Extract structured abstract information"""
        abstract_info = {
            'full_text': '',
            'sections': [],
            'has_content': False,
            'word_count': 0,
            'structured': False
        }
        
        all_text = []
        sections = []
        
        for abstract in article_xml.findall('.//Abstract'):
            for text_elem in abstract.findall('.//AbstractText'):
                label = text_elem.get('Label', '').strip()
                text = ''.join(text_elem.itertext()).strip()
                
                if text:
                    if label:
                        sections.append({'label': label, 'content': text})
                        all_text.append(f"{label}: {text}")
                        abstract_info['structured'] = True
                    else:
                        sections.append({'label': 'Main', 'content': text})
                        all_text.append(text)
        
        if all_text:
            abstract_info['full_text'] = '\n\n'.join(all_text)
            abstract_info['sections'] = sections
            abstract_info['has_content'] = True
            abstract_info['word_count'] = len(abstract_info['full_text'].split())
        
        return abstract_info
    
    def _extract_mesh_terms(self, article_xml) -> List[Dict]:
        """Extract MeSH terms"""
        mesh_terms = []
        
        for mesh_heading in article_xml.findall('.//MeshHeading'):
            descriptor = mesh_heading.find('DescriptorName')
            if descriptor is not None:
                mesh_info = {
                    'descriptor': {
                        'name': descriptor.text,
                        'ui': descriptor.get('UI', ''),
                        'major_topic': descriptor.get('MajorTopicYN', 'N') == 'Y'
                    },
                    'qualifiers': []
                }
                
                # Add qualifiers
                for qualifier in mesh_heading.findall('QualifierName'):
                    qualifier_info = {
                        'name': qualifier.text,
                        'ui': qualifier.get('UI', ''),
                        'major_topic': qualifier.get('MajorTopicYN', 'N') == 'Y'
                    }
                    mesh_info['qualifiers'].append(qualifier_info)
                
                mesh_terms.append(mesh_info)
        
        return mesh_terms
    
    def _extract_journal_info(self, article_xml) -> Dict:
        """Extract journal information"""
        journal = article_xml.find('.//Journal')
        if journal is None:
            return {}
        
        info = {}
        
        title = journal.find('.//Title')
        if title is not None:
            info['title'] = title.text
        
        iso_abbrev = journal.find('.//ISOAbbreviation')
        if iso_abbrev is not None:
            info['iso_abbreviation'] = iso_abbrev.text
        
        issn = journal.find('.//ISSN')
        if issn is not None:
            info['issn'] = issn.text
        
        return info
    
    def _extract_authors(self, article_xml) -> List[Dict]:
        """Extract author information"""
        authors = []
        author_list = article_xml.find('.//AuthorList')
        
        if author_list is not None:
            for author in author_list.findall('Author'):
                author_info = {}
                
                last_name = author.find('LastName')
                if last_name is not None:
                    author_info['last_name'] = last_name.text
                
                first_name = author.find('ForeName')
                if first_name is not None:
                    author_info['first_name'] = first_name.text
                
                if author_info:
                    authors.append(author_info)
        
        return authors
    
    def _extract_publication_info(self, article_xml) -> Dict:
        """Extract publication information"""
        pub_info = {}
        
        pub_date = article_xml.find('.//PubDate')
        if pub_date is not None:
            year = pub_date.find('Year')
            if year is not None:
                pub_info['year'] = int(year.text) if year.text.isdigit() else year.text
            
            month = pub_date.find('Month')
            if month is not None:
                pub_info['month'] = month.text
        
        return pub_info
    
    def _extract_identifiers(self, article_xml) -> Dict:
        """Extract article identifiers"""
        identifiers = {}
        
        for article_id in article_xml.findall('.//ArticleId'):
            id_type = article_id.get('IdType')
            if id_type and article_id.text:
                identifiers[id_type] = article_id.text
        
        return identifiers
    
    def _extract_keywords(self, article_xml) -> List[str]:
        """Extract keywords"""
        keywords = []
        
        for keyword_list in article_xml.findall('.//KeywordList'):
            for keyword in keyword_list.findall('Keyword'):
                if keyword.text:
                    keywords.append(keyword.text.strip())
        
        return keywords
    
    def _detect_language(self, text: str) -> str:
        """Detect language of abstract text"""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        # Spanish indicators
        spanish_indicators = [
            'objetivo', 'métodos', 'resultados', 'conclusión', 'conclusiones',
            'introducción', 'material', 'pacientes', 'estudio', 'análisis'
        ]
        
        # English indicators
        english_indicators = [
            'objective', 'methods', 'results', 'conclusion', 'conclusions',
            'introduction', 'material', 'patients', 'study', 'analysis'
        ]
        
        spanish_count = sum(1 for indicator in spanish_indicators if indicator in text_lower)
        english_count = sum(1 for indicator in english_indicators if indicator in text_lower)
        
        if spanish_count > english_count and spanish_count >= 2:
            return 'spanish'
        elif english_count > spanish_count and english_count >= 2:
            return 'english'
        else:
            return 'mixed_or_unknown'
    
    def _apply_final_filters(self, articles: List[Dict]) -> List[Dict]:
        """Apply final quality filters to articles"""
        filtered = []
        
        for article in articles:
            # Must have abstract
            if not article['abstract']['has_content']:
                continue
            
            # Must have MeSH terms
            if not article['mesh_terms']:
                continue
            
            # Abstract must have minimum length
            if article['abstract']['word_count'] < 50:
                continue
            
            filtered.append(article)
        
        return filtered
    
    def save_dataset(self, articles: List[Dict], filename: str):
        """Save the collected dataset to JSON file"""
        dataset = {
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'collection_method': 'Multi-strategy search with copyright and quality filters',
                'total_articles': len(articles),
                'filters_applied': [
                    'Spanish language',
                    'Has abstract (minimum 50 words)',
                    'Has MeSH terms',
                    'Copyright-free (open access or free full text)'
                ],
                'search_strategies': [
                    'Date-based search (2000-2025)',
                    'Open access journal search',
                    'Medical subject search'
                ]
            },
            'statistics': self._generate_statistics(articles),
            'articles': articles
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        self.logger.info(f"💾 Dataset saved to {filename}")
        self.logger.info(f"📁 File size: {file_size_mb:.1f} MB")
        self.logger.info(f"📊 Total articles: {len(articles):,}")
    
    def _generate_statistics(self, articles: List[Dict]) -> Dict:
        """Generate dataset statistics"""
        if not articles:
            return {}
        
        # Language distribution
        languages = [a['abstract_language'] for a in articles]
        language_counts = {lang: languages.count(lang) for lang in set(languages)}
        
        # Year distribution
        years = [a['publication_info'].get('year') for a in articles if a['publication_info'].get('year')]
        year_counts = {year: years.count(year) for year in set(years) if isinstance(year, int)}
        
        # Journal distribution
        journals = [a['journal'].get('title', 'Unknown') for a in articles]
        journal_counts = defaultdict(int)
        for journal in journals:
            journal_counts[journal] += 1
        
        return {
            'total_articles': len(articles),
            'language_distribution': language_counts,
            'year_range': {
                'earliest': min(y for y in years if isinstance(y, int)) if years else None,
                'latest': max(y for y in years if isinstance(y, int)) if years else None
            },
            'top_years': dict(sorted(year_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_journals': dict(sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            'avg_abstract_length': sum(a['abstract']['word_count'] for a in articles) / len(articles),
            'structured_abstracts': len([a for a in articles if a['abstract']['structured']]),
            'collection_stats': self.stats
        }
    
    def run_collection(self, target_articles: int = 50000, output_file: str = "spanish_medical_articles.json") -> List[Dict]:
        """
        Run the complete article collection process
        
        Args:
            target_articles: Target number of articles to collect
            output_file: Output filename for the dataset
            
        Returns:
            List of collected articles
        """
        self.logger.info("🚀 Starting Spanish Medical Article Collection")
        self.logger.info(f"Target: {target_articles:,} copyright-free articles with abstracts and MeSH terms")
        
        # Step 1: Collect PMIDs
        pmids = self.collect_comprehensive_dataset(target_articles)
        
        if not pmids:
            self.logger.error("❌ No articles found matching criteria")
            return []
        
        # Step 2: Fetch detailed article information
        articles = self.fetch_article_details(list(pmids))
        
        # Step 3: Save dataset
        self.save_dataset(articles, output_file)
        
        # Step 4: Print final summary
        self._print_summary(articles)
        
        return articles
    
    def _print_summary(self, articles: List[Dict]):
        """Print collection summary"""
        print("\n" + "="*80)
        print("📊 COLLECTION SUMMARY")
        print("="*80)
        print(f"✅ Total articles collected: {len(articles):,}")
        print(f"📋 Articles with abstracts: {len(articles):,} (100%)")
        print(f"🏷️  Articles with MeSH terms: {len(articles):,} (100%)")
        print(f"📖 Copyright-free articles: {len(articles):,} (100%)")
        
        if articles:
            # Language breakdown
            languages = [a['abstract_language'] for a in articles]
            spanish_count = languages.count('spanish')
            english_count = languages.count('english')
            
            print(f"\n🌍 Language Distribution:")
            print(f"   Spanish abstracts: {spanish_count:,}")
            print(f"   English abstracts: {english_count:,}")
            print(f"   Other/Mixed: {len(articles) - spanish_count - english_count:,}")
            
            # Year range
            years = [a['publication_info'].get('year') for a in articles if isinstance(a['publication_info'].get('year'), int)]
            if years:
                print(f"\n📅 Year Range: {min(years)} - {max(years)}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Collect copyright-free Spanish medical articles with abstracts and MeSH terms'
    )
    
    parser.add_argument(
        '--target', 
        type=int, 
        default=50000,
        help='Target number of articles to collect (default: 50,000)'
    )
    
    parser.add_argument(
        '--output',
        default='spanish_medical_articles.json',
        help='Output filename for the dataset (default: spanish_medical_articles.json)'
    )
    
    parser.add_argument(
        '--api-key',
        help='NCBI API key for higher rate limits (recommended)'
    )
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = SpanishArticleCollector(api_key=args.api_key)
    
    # Run collection
    articles = collector.run_collection(
        target_articles=args.target,
        output_file=args.output
    )
    
    if articles:
        print(f"\n🎉 Collection completed successfully!")
        print(f"📁 Dataset saved to: {args.output}")
    else:
        print(f"\n❌ Collection failed - no articles found")


if __name__ == "__main__":
    main()