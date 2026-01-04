"""
Cybersecurity URL Crawler

A tool for crawling and filtering cybersecurity-related content from GitHub repositories
and their linked URLs using LLM-based relevance classification.

Requirements:
    pip install gitingest requests openai tqdm
"""

import argparse
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import urlparse

import requests
from gitingest import ingest
from openai import OpenAI

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# Configuration
@dataclass
class CrawlerConfig:
    """Configuration for the cybersecurity crawler."""
    
    # Target repository
    target_repo_url: str = "https://github.com/ramimac/aws-customer-security-incidents"
    
    # Crawling settings
    crawl_depth: int = 4
    max_workers: int = 2
    crawl_api_base_url: str = "https://crawl4ai.infra.pubkeypair.com/md"
    
    # LLM settings
    llm_base_url: str = "http://192.168.0.194:8000/v1"
    llm_api_key: str = "fake-key"
    llm_model: str = "gpt-oss-20b"
    llm_temperature: float = 0.0
    
    # Output settings
    output_file: Path = Path("dataset.md")
    
    # UI settings
    show_progress: bool = False
    log_level: str = "INFO"
    
    # Context for relevance filtering
    crawl_context: str = (
        "Cybersecurity related topics such as security incidents, vulnerabilities, "
        "exploits, threat intelligence, security best practices, incident response, "
        "security frameworks, compliance, and security tools."
    )
    
    # Prompt template for LLM classification
    prompt_template: str = """You are an expert in cybersecurity and information retrieval. Your task is to determine if a given URL is related to cybersecurity topics.

Context: {context}

Please analyze the URL and respond with only 'True' or 'False' indicating whether it's relevant to cybersecurity.

URL: {url}"""


# Regular expressions
URL_REGEX = re.compile(r'https?://[^\s]+')
URL_CLEANUP_REGEX = re.compile(r'\){0,1}(,){0,1}(\n){0,1}$')


# Setup logging
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return a logger instance."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class ProgressTracker:
    """Handles progress tracking for both CLI and library usage."""
    
    def __init__(self, enabled: bool = False, total: int = 0, desc: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            enabled: Whether to show progress bars
            total: Total number of items to process
            desc: Description for the progress bar
        """
        self.enabled = enabled and TQDM_AVAILABLE
        self.pbar = None
        self.lock = threading.Lock()
        
        if self.enabled:
            self.pbar = tqdm(total=total, desc=desc, unit="url", colour="green")
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        if self.enabled and self.pbar:
            with self.lock:
                self.pbar.update(n)
    
    def set_description(self, desc: str):
        """Update the progress bar description."""
        if self.enabled and self.pbar:
            with self.lock:
                self.pbar.set_description(desc)
    
    def close(self):
        """Close the progress bar."""
        if self.enabled and self.pbar:
            self.pbar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CybersecurityCrawler:
    """Main crawler class for extracting and filtering cybersecurity content."""
    
    def __init__(self, config: CrawlerConfig):
        """Initialize the crawler with given configuration."""
        self.config = config
        self.crawled_urls: Set[str] = set()
        self.crawled_urls_lock = threading.Lock()
        self.all_responses: List[str] = []
        self.responses_lock = threading.Lock()
        
        # Update logger level
        global logger
        logger = setup_logging(config.log_level)
        
        # Initialize OpenAI client
        self.llm_client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key
        )
        
        # Progress tracking
        self.main_progress: Optional[ProgressTracker] = None
        self.depth_progress: Optional[ProgressTracker] = None
        
        logger.info("Cybersecurity crawler initialized")
    
    @staticmethod
    def clean_url(url: str) -> str:
        """Remove trailing punctuation and newlines from URLs."""
        return URL_CLEANUP_REGEX.sub('', url)
    
    def extract_urls_from_text(self, text: str) -> List[str]:
        """Extract and clean URLs from text content."""
        raw_urls = URL_REGEX.findall(text)
        return [self.clean_url(url) for url in raw_urls]
    
    def crawl_url(self, url: str) -> dict:
        """
        Crawl a single URL using the Crawl4AI service.
        
        Returns:
            Dictionary with markdown content or empty dict on failure.
        """
        with self.crawled_urls_lock:
            if url in self.crawled_urls:
                logger.debug(f"URL already crawled, skipping: {url}")
                return {'markdown': ''}
            self.crawled_urls.add(url)
        
        try:
            logger.debug(f"Crawling URL: {url}")
            response = requests.post(
                self.config.crawl_api_base_url,
                json={"url": url, "f": "raw", "q": None, "c": "0"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to crawl URL {url}: {e}")
            return {'markdown': ''}
        except Exception as e:
            logger.error(f"Unexpected error crawling URL {url}: {e}")
            return {'markdown': ''}
    
    def is_url_relevant(self, url: str) -> bool:
        """
        Check if a URL is relevant to cybersecurity using LLM classification.
        
        Args:
            url: The URL to classify
            
        Returns:
            True if relevant, False otherwise
        """
        try:
            prompt = self.config.prompt_template.format(
                context=self.config.crawl_context,
                url=url
            )
            
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
            )
            
            result = response.choices[0].message.content.strip().lower()
            return result == 'true'
            
        except Exception as e:
            logger.warning(f"LLM classification failed for {url}: {e}. Defaulting to relevant.")
            return True
    
    def filter_urls_by_relevance(self, urls: List[str]) -> List[str]:
        """Filter URLs based on cybersecurity relevance using LLM."""
        filtered_urls = []
        
        with ProgressTracker(
            enabled=self.config.show_progress,
            total=len(urls),
            desc="Filtering URLs"
        ) as filter_progress:
            for url in urls:
                try:
                    is_relevant = self.is_url_relevant(url)
                    if is_relevant:
                        logger.info(f"✓ RELEVANT: {url}")
                        filtered_urls.append(url)
                    else:
                        logger.info(f"✗ IRRELEVANT: {url}")
                except Exception as e:
                    logger.error(f"Error filtering URL {url}: {e}")
                finally:
                    filter_progress.update(1)
        
        return filtered_urls
    
    def crawl_article_in_depth(self, url: str) -> List[str]:
        """
        Recursively crawl a URL and its linked pages to specified depth.
        
        Args:
            url: Starting URL to crawl
            
        Returns:
            List of markdown responses from all crawled pages
        """
        responses = []
        urls_to_crawl = [url]
        
        for depth in range(self.config.crawl_depth):
            logger.info(f"[Depth {depth + 1}/{self.config.crawl_depth}] "
                       f"Crawling {len(urls_to_crawl)} URLs")
            
            current_urls = urls_to_crawl.copy()
            next_urls = []
            
            with ProgressTracker(
                enabled=self.config.show_progress,
                total=len(current_urls),
                desc=f"Depth {depth + 1}/{self.config.crawl_depth}"
            ) as depth_progress:
                for idx, current_url in enumerate(current_urls, 1):
                    depth_progress.set_description(
                        f"Depth {depth + 1}/{self.config.crawl_depth} - URL {idx}/{len(current_urls)}"
                    )
                    
                    # Crawl current URL
                    response_data = self.crawl_url(current_url)
                    markdown_content = response_data.get('markdown', '')
                    
                    if markdown_content:
                        formatted_response = f"Source: {current_url}\n\n\n{markdown_content}"
                        responses.append(formatted_response)
                        
                        # Extract and filter sub-URLs
                        sub_urls = self.extract_urls_from_text(markdown_content)
                        sub_urls_unique = list(set(sub_urls))
                        
                        logger.debug(f"Found {len(sub_urls_unique)} unique URLs in {current_url}")
                        
                        filtered_sub_urls = self.filter_urls_by_relevance(sub_urls_unique)
                        next_urls.extend(filtered_sub_urls)
                        
                        logger.info(f"[Depth {depth + 1}/{self.config.crawl_depth}] "
                                   f"[URL {idx}/{len(current_urls)}] "
                                   f"Crawled {current_url}, found {len(filtered_sub_urls)} relevant sub-URLs")
                    
                    depth_progress.update(1)
            
            # Prepare for next depth
            urls_to_crawl = list(set(next_urls))
            logger.info(f"[Depth {depth + 1}/{self.config.crawl_depth}] "
                       f"Completed. Found {len(urls_to_crawl)} unique relevant URLs for next depth")
        
        return responses
    
    def save_responses(self):
        """Save all collected responses to output file."""
        try:
            with self.responses_lock:
                content = '\n\n---\n\n'.join(self.all_responses)
                self.config.output_file.write_text(content, encoding='utf-8')
                logger.info(f"Saved {len(self.all_responses)} responses to {self.config.output_file}")
        except Exception as e:
            logger.error(f"Failed to save responses: {e}")
    
    def crawl_and_collect(self, index: int, url: str):
        """
        Crawl a URL and collect responses (thread-safe).
        
        Args:
            index: Index of the URL in the processing queue
            url: URL to crawl
        """
        try:
            logger.info(f"Starting crawl for URL {index + 1}: {url}")
            responses = self.crawl_article_in_depth(url)
            
            with self.responses_lock:
                self.all_responses.extend(responses)
                self.save_responses()
                logger.info(f"Progress: {index + 1} URLs processed")
                
            if self.main_progress:
                self.main_progress.update(1)
                
        except Exception as e:
            logger.error(f"Failed to crawl and collect URL {url}: {e}")
            if self.main_progress:
                self.main_progress.update(1)
    
    def run(self):
        """Main execution method to run the crawler."""
        logger.info(f"Starting crawl of repository: {self.config.target_repo_url}")
        
        # Ingest repository content
        try:
            if self.config.show_progress:
                print("📥 Ingesting repository content...")
            summary, tree, content = ingest(self.config.target_repo_url)
            logger.info(f"Repository ingested successfully")
            logger.info(f"Summary: {summary}")
        except Exception as e:
            logger.error(f"Failed to ingest repository: {e}")
            return
        
        # Extract URLs from repository content
        urls_to_process = self.extract_urls_from_text(content)
        logger.info(f"Found {len(urls_to_process)} URLs in repository")
        
        if not urls_to_process:
            logger.warning("No URLs found in repository content")
            return
        
        # Process URLs concurrently
        if self.config.show_progress:
            print(f"\n🔍 Processing {len(urls_to_process)} URLs from repository...")
        
        self.main_progress = ProgressTracker(
            enabled=self.config.show_progress,
            total=len(urls_to_process),
            desc="Main Progress"
        )
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(self.crawl_and_collect, idx, url)
                for idx, url in enumerate(urls_to_process)
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in concurrent execution: {e}")
        
        self.main_progress.close()
        
        logger.info(f"Crawling completed. Total responses collected: {len(self.all_responses)}")
        logger.info(f"Results saved to: {self.config.output_file}")
        
        if self.config.show_progress:
            print(f"\n✅ Crawling completed!")
            print(f"📊 Total responses collected: {len(self.all_responses)}")
            print(f"💾 Results saved to: {self.config.output_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cybersecurity URL Crawler - Extract and filter cybersecurity content from repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python crawler.py
  
  # Crawl a specific repository with progress bar
  python crawler.py --repo https://github.com/user/repo --progress
  
  # Custom crawl depth and parallel workers
  python crawler.py --depth 5 --workers 4 --progress
  
  # Use custom LLM endpoint
  python crawler.py --llm-url http://localhost:8000/v1 --llm-model gpt-4
  
  # Enable debug logging
  python crawler.py --log-level DEBUG --progress
        """
    )
    
    # Repository settings
    parser.add_argument(
        "--repo",
        type=str,
        default="https://github.com/ramimac/aws-customer-security-incidents",
        help="GitHub repository URL to crawl (default: aws-customer-security-incidents)"
    )
    
    # Crawling settings
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Maximum crawl depth for recursive URL following (default: 4)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel worker threads (default: 2)"
    )
    
    parser.add_argument(
        "--crawl-api",
        type=str,
        default="https://crawl4ai.infra.pubkeypair.com/md",
        help="Crawl4AI API base URL (default: https://crawl4ai.infra.pubkeypair.com/md)"
    )
    
    # LLM settings
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://192.168.0.194:8000/v1",
        help="LLM API base URL (default: http://192.168.0.194:8000/v1)"
    )
    
    parser.add_argument(
        "--llm-key",
        type=str,
        default="fake-key",
        help="LLM API key (default: fake-key)"
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-oss-20b",
        help="LLM model name (default: gpt-oss-20b)"
    )
    
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="LLM temperature for classification (default: 0.0)"
    )
    
    # Output settings
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="cybersecurity_dataset.md",
        help="Output file path (default: cybersecurity_dataset.md)"
    )
    
    # UI settings
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show colored progress bars (requires tqdm)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # Context settings
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Custom context for cybersecurity relevance filtering"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the crawler."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Check if tqdm is available when progress is requested
    if args.progress and not TQDM_AVAILABLE:
        print("⚠️  Warning: --progress flag requires 'tqdm' package. Install with: pip install tqdm")
        print("Continuing without progress bars...\n")
        args.progress = False
    
    # Create configuration from arguments
    config = CrawlerConfig(
        target_repo_url=args.repo,
        crawl_depth=args.depth,
        max_workers=args.workers,
        crawl_api_base_url=args.crawl_api,
        llm_base_url=args.llm_url,
        llm_api_key=args.llm_key,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        output_file=Path(args.output),
        show_progress=args.progress,
        log_level=args.log_level,
    )
    
    # Update context if provided
    if args.context:
        config.crawl_context = args.context
    
    # Display configuration if in CLI mode
    if args.progress:
        print("🚀 Cybersecurity URL Crawler")
        print("=" * 60)
        print(f"Repository:     {config.target_repo_url}")
        print(f"Crawl Depth:    {config.crawl_depth}")
        print(f"Workers:        {config.max_workers}")
        print(f"LLM Endpoint:   {config.llm_base_url}")
        print(f"LLM Model:      {config.llm_model}")
        print(f"Output File:    {config.output_file}")
        print(f"Log Level:      {config.log_level}")
        print("=" * 60)
        print()
    
    # Initialize and run crawler
    try:
        crawler = CybersecurityCrawler(config)
        crawler.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Crawl interrupted by user. Partial results may be saved.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()